"""Phase 1.5: data integrity verification of the local PTB-XL v1.0.3 copy (criteria C1-C9).

Fold 10 (sealed test set) gets checksum, file-name and header checks only; its
signal samples are never read. See reports/phase1_5_data/CRITERIA.md.

    python research/phase1_5_verify_data.py --data PTB-XL --out reports/phase1_5_data
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import time
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_RECORDS = 21_799
EXPECTED_PATIENTS = 18_869
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUMS_SOURCES = {
    "physionet.org": "https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt",
    "s3": "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/SHA256SUMS.txt",
}
REQUIRED = ("ptbxl_database.csv", "scp_statements.csv", "RECORDS", "LICENSE.txt", "SHA256SUMS.txt")


def load_downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_ptbxl.py"
    spec = importlib.util.spec_from_file_location("download_ptbxl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dropped_ids(changelog: Path) -> set[int]:
    """ecg_ids removed as duplicates in v1.0.3, from the official changelog."""
    for line in changelog.read_text().splitlines():
        if line.startswith("Dropped ecg_ids"):
            return set(ast.literal_eval(line.split("=", 1)[1].strip()))
    raise ValueError("no 'Dropped ecg_ids' line in changelog")


def file_status(task: tuple[Path, str]) -> str:
    path, expected = task
    if not path.exists():
        return "missing"
    return "ok" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "corrupt"


def check_header(task: tuple[int, str, int]) -> dict:
    """C7 for every record; C8 (sample-level) only when read_signal is true (folds 1-9)."""
    import wfdb

    ecg_id, base, read_signal = task
    out: dict = {"ecg_id": ecg_id, "problems": []}
    h = wfdb.rdheader(base)
    expected_name = Path(base).name
    checks = {
        "record_name": h.record_name == expected_name,
        "fs": h.fs == 100,
        "n_sig": h.n_sig == 12,
        "sig_len": h.sig_len == 1000,
        "units": list(h.units) == ["mV"] * 12,
        "lead_order": [s.upper() for s in h.sig_name] == LEADS,
        "fmt": list(h.fmt) == ["16"] * 12,
        "file_name": list(h.file_name) == [f"{expected_name}.dat"] * 12,
    }
    out["problems"] = [k for k, ok in checks.items() if not ok]
    out["adc_gain"] = sorted(set(h.adc_gain))
    if read_signal:
        rec = wfdb.rdrecord(base, physical=False)
        d = rec.d_signal.astype(np.int64)
        # WFDB checksum: 16-bit sum of each signal's samples (compare modulo 2**16, since
        # headers may store it signed or unsigned).
        calc = d.sum(axis=0)
        if any((int(c) - int(s)) % 65536 for c, s in zip(calc, h.checksum, strict=True)):
            out["problems"].append("wfdb_checksum")
        if d.shape != (1000, 12):
            out["problems"].append("signal_shape")
        p = (d - np.asarray(h.baseline)) / np.asarray(h.adc_gain)
        if not np.isfinite(p).all():
            out["problems"].append("non_finite")
        out["adc_min"], out["adc_max"] = int(d.min()), int(d.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PTB-XL"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase1_5_data"))
    ap.add_argument("--workers", type=int, default=38)
    args = ap.parse_args()
    root: Path = args.data
    t0 = time.time()
    res: dict = {}

    # C1 required files
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if not (root / "records100").is_dir():
        missing.append("records100/")
    res["C1_required_files"] = {"pass": not missing, "missing": missing}

    # C2 checksum list identical on independent sources (redirects are not followed)
    local = (root / "SHA256SUMS.txt").read_bytes()
    sources = {"local": hashlib.sha256(local).hexdigest()}
    for name, url in SUMS_SOURCES.items():
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(url, timeout=60) as resp:
                sources[name] = hashlib.sha256(resp.read()).hexdigest()
        except Exception as exc:
            sources[name] = f"unavailable: {exc}"
    res["C2_checksum_list_sources"] = {"pass": len(set(sources.values())) == 1, **sources}

    # C3 all listed files present with correct SHA-256
    dl = load_downloader()
    manifest = {}
    for line in local.decode().splitlines():
        digest, rel = line.split(maxsplit=1)
        if dl.wanted(rel, (100,)):
            manifest[rel] = digest
    with Pool(args.workers) as pool:
        statuses = pool.map(file_status, [(root / r, d) for r, d in manifest.items()], 256)
    status_counts = Counter(statuses)
    bad = [rel for rel, s in zip(manifest, statuses, strict=True) if s != "ok"]
    res["C3_checksums"] = {
        "pass": not bad,
        "files_checked": len(manifest),
        "status": dict(status_counts),
        "bad": bad[:20],
    }

    # C4 no unexpected files under records100/
    on_disk = {
        p.relative_to(root).as_posix() for p in (root / "records100").rglob("*") if p.is_file()
    }
    listed = {r for r in manifest if r.startswith("records100/")}
    extra = sorted(on_disk - listed)
    res["C4_unexpected_files"] = {
        "pass": not extra,
        "n_files_on_disk": len(on_disk),
        "extra": extra[:20],
    }

    # C5 counts
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    n_rec, n_pat = len(db), db.patient_id.nunique()
    res["C5_counts"] = {
        "pass": n_rec == EXPECTED_RECORDS and n_pat == EXPECTED_PATIENTS,
        "records": n_rec,
        "patients": n_pat,
        "expected": [EXPECTED_RECORDS, EXPECTED_PATIENTS],
    }

    # C6 folder structure: one .hea + one .dat per record at the expected path
    expected_lr = db.index.map(lambda i: f"records100/{i // 1000 * 1000:05d}/{i:05d}_lr")
    lr_ok = (db.filename_lr == expected_lr).all()
    missing_pair = [
        f for f in db.filename_lr if f"{f}.hea" not in on_disk or f"{f}.dat" not in on_disk
    ]
    record_files = {p.rsplit(".", 1)[0] for p in on_disk}
    orphans = sorted(record_files - set(db.filename_lr))
    folder_sizes = Counter(Path(f).parent.name for f in db.filename_lr)
    res["C6_structure"] = {
        "pass": bool(lr_ok)
        and not missing_pair
        and not orphans
        and max(folder_sizes.values()) <= 1000,
        "filename_lr_matches_ecg_id": bool(lr_ok),
        "records_missing_hea_or_dat": missing_pair[:20],
        "orphan_record_files": orphans[:20],
        "n_folders": len(folder_sizes),
        "max_records_per_folder": max(folder_sizes.values()),
    }

    # C7 headers (all folds) + C8 sample-level checks (folds 1-9 only)
    tasks = [(int(i), str(root / f), int(fold <= 9)) for i, f, fold in
             zip(db.index, db.filename_lr, db.strat_fold, strict=True)]  # fmt: skip
    with Pool(args.workers) as pool:
        checked = pool.map(check_header, tasks, chunksize=64)
    hdr_problems = Counter(p for c in checked for p in c["problems"] if p in {
        "record_name", "fs", "n_sig", "sig_len", "units", "lead_order", "fmt", "file_name"})  # fmt: skip
    sig_problems = Counter(p for c in checked for p in c["problems"] if p in {
        "wfdb_checksum", "signal_shape", "non_finite"})  # fmt: skip
    gains = Counter(g for c in checked for g in c["adc_gain"])
    res["C7_headers"] = {
        "pass": not hdr_problems,
        "headers_checked": len(checked),
        "problems": dict(hdr_problems),
        "adc_gain_values": {str(k): v for k, v in gains.items()},
        "lead_names_as_stored": "I II III AVR AVL AVF V1-V6 (upper case)",
    }
    sig = [c for c in checked if "adc_min" in c]
    adc_min, adc_max = min(c["adc_min"] for c in sig), max(c["adc_max"] for c in sig)
    res["C8_signals_folds_1_9"] = {
        "pass": not sig_problems,
        "records_read": len(sig),
        "problems": dict(sig_problems),
        "failing_ecg_ids": [c["ecg_id"] for c in sig if c["problems"]][:20],
        "adc_range": [adc_min, adc_max],
        "records_touching_int16_limits": sum(
            c["adc_min"] <= -32768 or c["adc_max"] >= 32767 for c in sig
        ),
    }

    # C9 metadata consistency
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    codes = db.scp_codes.apply(ast.literal_eval)
    undefined = sorted({k for c in codes for k in c} - set(scp.index))
    records_file = set((root / "RECORDS").read_text().split())
    rec100 = {r for r in records_file if r.startswith("records100/")}
    folds_per_patient = db.groupby("patient_id").strat_fold.nunique()
    dropped = dropped_ids(root / "ptbxl_v103_changelog.txt")
    full_range = set(range(1, db.index.max() + 1))
    checks9 = {
        "ecg_id_unique": bool(db.index.is_unique),
        "strat_fold_in_1_10": bool(db.strat_fold.between(1, 10).all()),
        "all_scp_codes_defined": not undefined,
        "RECORDS_matches_database": rec100 == set(db.filename_lr),
        "patient_in_single_fold": bool((folds_per_patient == 1).all()),
        "no_v103_dropped_duplicates_present": not (dropped & set(db.index)),
        "ecg_ids_are_1_to_max_minus_dropped": set(db.index) == full_range - dropped,
    }
    res["C9_metadata"] = {
        "pass": all(checks9.values()),
        **checks9,
        "undefined_scp_codes": undefined,
        "n_dropped_ids_in_changelog": len(dropped),
        "RECORDS_entries": len(records_file),
        # Diagnostic: entries that fuse two record paths because a newline is missing.
        "RECORDS_fused_entries": sorted(r for r in records_file if r.count("records") > 1),
        "RECORDS_matches_database_after_splitting_fused": {
            p
            for r in records_file
            for p in re.findall(r"records\d+/\d{5}/\d{5}_(?:lr|hr)", r)
            if p.startswith("records100/")
        }
        == set(db.filename_lr),
        "patients_in_multiple_folds": int((folds_per_patient > 1).sum()),
    }

    # Descriptive dataset version record (committed; referenced by later phases).
    size = sum((root / r).stat().st_size for r in manifest)
    res["dataset_record"] = {
        "dataset": "PTB-XL",
        "version": "1.0.3",
        "doi": "10.13026/kfzx-aw45",
        "sha256sums_sha256": sources["local"],
        "files_used": len(manifest),
        "bytes_used": size,
        "records_per_fold": {
            int(k): int(v) for k, v in db.strat_fold.value_counts().sort_index().items()
        },
        "patients_per_fold": {
            int(k): int(v) for k, v in db.groupby("strat_fold").patient_id.nunique().items()
        },
    }
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("C"))
    res["runtime_s"] = round(time.time() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "integrity.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


if __name__ == "__main__":
    main()
