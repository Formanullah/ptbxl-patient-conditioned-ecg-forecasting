import hashlib
import http.server
import importlib.util
import sys
import threading
from collections import Counter
from functools import partial
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_ptbxl.py"
spec = importlib.util.spec_from_file_location("download_ptbxl", SCRIPT)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

FILES = {
    "ptbxl_database.csv": b"ecg_id,patient_id\n1,10\n",
    "records100/00000/00001_lr.hea": b"00001_lr 12 100 1000\n",
    "records100/00000/00001_lr.dat": bytes(range(256)) * 4,
    "records500/00000/00001_hr.dat": b"not wanted at 100 Hz",
}


def make_mirror(root: Path, corrupt: str | None = None) -> None:
    lines = []
    for rel, data in FILES.items():
        lines.append(f"{hashlib.sha256(data).hexdigest()} {rel}")
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"corrupted" if rel == corrupt else data)
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture
def serve(tmp_path, monkeypatch):
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(var, "*")
    servers = []

    def start(
        name: str,
        corrupt: str | None = None,
        keep_alive: bool = False,
        ports=None,
        redirect_to: str | None = None,
    ) -> str:
        root = tmp_path / name
        make_mirror(root, corrupt)

        class Handler(http.server.SimpleHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if keep_alive else "HTTP/1.0"

            def do_GET(self):
                if redirect_to is None:
                    return super().do_GET()
                self.send_response(302)
                self.send_header("Location", redirect_to + self.path.lstrip("/"))
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                if ports is not None:
                    ports.append(self.client_address[1])

        handler = partial(Handler, directory=str(root))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/"

    yield start
    for server in servers:
        server.shutdown()


def run(monkeypatch, capsys, *args: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["download_ptbxl.py", *args])
    code = dl.main()
    return code, capsys.readouterr().out


def test_download_verify_and_skip(tmp_path, serve, monkeypatch, capsys):
    mirror, dest = serve("good"), tmp_path / "data"
    code, out = run(monkeypatch, capsys, "--root", str(dest), "--mirror", mirror)
    assert code == 0 and '"downloaded": 3' in out
    assert not (dest / "records500").exists()
    code, out = run(monkeypatch, capsys, "--root", str(dest), "--mirror", mirror)
    assert code == 0 and '"ok": 3' in out and "downloaded" not in out


def test_corrupted_local_file_is_replaced(tmp_path, serve, monkeypatch, capsys):
    mirror, dest = serve("good"), tmp_path / "data"
    run(monkeypatch, capsys, "--root", str(dest), "--mirror", mirror)
    target = dest / "records100/00000/00001_lr.dat"
    target.write_bytes(b"bit rot")
    code, out = run(monkeypatch, capsys, "--root", str(dest), "--verify-only")
    assert code == 1 and "corrupt: records100/00000/00001_lr.dat" in out
    code, out = run(monkeypatch, capsys, "--root", str(dest), "--mirror", mirror)
    assert code == 0 and '"downloaded": 1' in out
    assert target.read_bytes() == FILES["records100/00000/00001_lr.dat"]


def test_corrupted_mirror_falls_back_to_next(tmp_path, serve, monkeypatch, capsys):
    bad = serve("bad", corrupt="records100/00000/00001_lr.dat")
    good = serve("good")
    dest = tmp_path / "data"
    code, _ = run(monkeypatch, capsys, "--root", str(dest), "--mirror", bad, "--mirror", good)
    assert code == 0
    assert (dest / "records100/00000/00001_lr.dat").read_bytes() == FILES[
        "records100/00000/00001_lr.dat"
    ]


def test_unrecoverable_file_is_reported_not_fatal(tmp_path, serve, monkeypatch, capsys):
    bad = serve("bad", corrupt="records100/00000/00001_lr.dat")
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    code, out = run(monkeypatch, capsys, "--root", str(tmp_path / "d"), "--mirror", bad)
    assert code == 1 and "failed: records100/00000/00001_lr.dat" in out
    assert '"downloaded": 2' in out


def test_redirects_are_followed(tmp_path, serve, monkeypatch, capsys):
    target = serve("target")
    front = serve("front", redirect_to=target)
    code, out = run(monkeypatch, capsys, "--root", str(tmp_path / "d"), "--mirror", front)
    assert code == 0 and '"downloaded": 3' in out


def test_connection_is_reused(tmp_path, serve, monkeypatch, capsys):
    ports: list[int] = []
    mirror = serve("good", keep_alive=True, ports=ports)
    monkeypatch.setattr(dl, "_local", threading.local())
    code, _ = run(monkeypatch, capsys, "--root", str(tmp_path / "d"), "--mirror", mirror,
                  "--workers", "1")  # fmt: skip
    # Main thread fetches the checksum list; the single worker fetches 3 files over 1 connection.
    assert code == 0 and sorted(Counter(ports).values()) == [1, 3]
