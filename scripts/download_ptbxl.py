"""Download and verify the PTB-XL v1.0.3 files needed for 100 Hz work.

Standard library only, so it runs unchanged on Windows, Linux and macOS without
admin rights. Files are fetched from the public PhysioNet S3 mirror (falling back
to physionet.org), verified against the official SHA256SUMS.txt, and skipped when
an existing local copy already matches. Re-running resumes an interrupted download.

    python scripts/download_ptbxl.py --root PTB-XL
    python scripts/download_ptbxl.py --root PTB-XL --verify-only
    python scripts/download_ptbxl.py --root sample --include records100/00000/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/",
    "https://physionet.org/files/ptb-xl/1.0.3/",
)
METADATA = (
    "ptbxl_database.csv",
    "scp_statements.csv",
    "LICENSE.txt",
    "RECORDS",
    "ptbxl_v102_changelog.txt",
    "ptbxl_v103_changelog.txt",
)
USER_AGENT = "ptbxl-forecast-downloader/1.0"

_local = threading.local()
_ssl_context = ssl.create_default_context()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connection(url: urllib.parse.SplitResult) -> http.client.HTTPConnection:
    """Persistent per-thread connection, tunnelled through the system proxy if one applies.

    Reusing the connection avoids one TLS handshake per file (measured: 3-17x faster
    than a new connection per request, as urllib does).
    """
    conns = _local.__dict__.setdefault("conns", {})
    key = (url.scheme, url.netloc)
    if key not in conns:
        https = url.scheme == "https"
        cls = http.client.HTTPSConnection if https else http.client.HTTPConnection
        extra = {"context": _ssl_context} if https else {}
        proxy = urllib.request.getproxies().get(url.scheme)
        if proxy and not urllib.request.proxy_bypass(url.hostname):
            p = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
            headers = {}
            if p.username:
                cred = (
                    f"{urllib.parse.unquote(p.username)}:{urllib.parse.unquote(p.password or '')}"
                )
                headers["Proxy-Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
            conn = cls(p.hostname, p.port or 8080, timeout=60, **extra)
            conn.set_tunnel(url.hostname, url.port, headers=headers)
        else:
            conn = cls(url.hostname, url.port, timeout=60, **extra)
        conns[key] = conn
    return conns[key]


def _get(url: str, redirects: int = 5) -> bytes:
    parts = urllib.parse.urlsplit(url)
    conn = _connection(parts)
    for attempt in range(2):  # the server may have closed an idle kept-alive connection
        try:
            conn.request("GET", parts.path, headers={"User-Agent": USER_AGENT})
            resp = conn.getresponse()
            data = resp.read()
            break
        except (http.client.HTTPException, OSError):
            conn.close()  # the next request reconnects
            if attempt:
                raise
    location = resp.getheader("Location")
    if resp.status in (301, 302, 303, 307, 308) and location and redirects > 0:
        # physionet.org redirects file downloads to its S3 mirror (observed 2026-11-22).
        return _get(urllib.parse.urljoin(url, location), redirects - 1)
    if resp.status != 200:
        raise OSError(f"HTTP {resp.status} for {url}")
    return data


def fetch(
    rel: str,
    dest: Path,
    mirrors: tuple[str, ...] = MIRRORS,
    expected: str | None = None,
    retries: int = 4,
) -> None:
    """Download ``rel`` atomically; with ``expected`` set, accept only a matching SHA-256.

    A failed or corrupted download is retried on the next mirror, then after back-off.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: object = None
    for attempt in range(retries):
        for mirror in mirrors:
            try:
                data = _get(mirror + urllib.parse.quote(rel))
            except Exception as exc:  # network errors vary by platform
                last_error = exc
                continue
            if expected is not None and hashlib.sha256(data).hexdigest() != expected:
                last_error = f"checksum mismatch from {mirror}"
                continue
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return
        time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {rel}: {last_error}")


def wanted(rel: str, rates: tuple[int, ...], include: tuple[str, ...] = ()) -> bool:
    if include:
        return rel.startswith(include)
    if rel in METADATA:
        return True
    return any(rel.startswith(f"records{rate}/") for rate in rates)


def ensure(
    rel: str, expected: str, root: Path, verify_only: bool, mirrors: tuple[str, ...] = MIRRORS
) -> str:
    path = root / rel
    if path.exists() and sha256(path) == expected:
        return "ok"
    if verify_only:
        return "missing" if not path.exists() else "corrupt"
    try:
        fetch(rel, path, mirrors, expected)
    except RuntimeError:
        return "failed"
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("PTB-XL"))
    parser.add_argument("--rates", type=int, nargs="+", default=[100], choices=[100, 500])
    parser.add_argument(
        "--include", action="append", default=[], help="only paths with this prefix (repeatable)"
    )
    parser.add_argument(
        "--mirror", action="append", default=[], help="base URL ending in '/' (repeatable)"
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root: Path = args.root
    mirrors = tuple(args.mirror) or MIRRORS
    root.mkdir(parents=True, exist_ok=True)
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        fetch("SHA256SUMS.txt", sums, mirrors)
    manifest = {}
    for line in sums.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        if wanted(rel, tuple(args.rates), tuple(args.include)):
            manifest[rel] = digest

    counts: dict[str, int] = {}
    problems: list[str] = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(ensure, rel, digest, root, args.verify_only, mirrors): rel
            for rel, digest in manifest.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if status in {"missing", "corrupt", "failed"}:
                problems.append(f"{status}: {futures[future]}")
            if done % 5000 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} {counts}", flush=True)

    report = {
        "root": str(root.resolve()),
        "files_expected": len(manifest),
        "status": counts,
        "problems": problems[:50],
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
