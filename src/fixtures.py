"""Small synthetic WFDB fixtures for software checks, never research evidence."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from config import LEADS, file_hash, load_config


def synthetic_dataset(root, patients=(4, 2, 2), seed=42):
    root = Path(root)
    folder = root / "records100" / "00000"
    folder.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    time = np.arange(1000) / 100
    index, patient = 1, 1
    for split, count in zip((1, 9, 10), patients, strict=True):
        for _ in range(count):
            for record_number in range(2):
                wave = np.column_stack(
                    [
                        0.1 * np.sin(2 * np.pi * (1 + patient / 50) * time + lead / 7)
                        + 0.01 * rng.normal(size=1000)
                        for lead in range(12)
                    ]
                )
                name = f"{index:05d}_lr"
                wfdb.wrsamp(
                    name,
                    fs=100,
                    units=["mV"] * 12,
                    sig_name=LEADS,
                    p_signal=wave,
                    fmt=["16"] * 12,
                    write_dir=str(folder),
                )
                rows.append(
                    {
                        "ecg_id": index,
                        "patient_id": patient,
                        "strat_fold": split,
                        "filename_lr": f"records100/00000/{name}",
                        "age": 300 if patient == 1 else 30 + patient,
                        "sex": patient % 2,
                        "recording_date": f"1990-01-{record_number + 1:02d} 12:00:00",
                        "scp_codes": "{'NORM': 100}",
                        "height": 170 + patient % 3,
                        "weight": 70 + patient % 4,
                    }
                )
                index += 1
            patient += 1
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame({"code": ["NORM"], "diagnostic": [1], "diagnostic_class": ["NORM"]}).set_index(
        "code"
    ).to_csv(root / "scp_statements.csv")
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{file_hash(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    return root


def fixture_config(work, base=None):
    work = Path(work)
    cfg = copy.deepcopy(base) if base else load_config()
    cfg["study_id"] = "synthetic_software_validation"
    cfg["data"]["root"] = str(synthetic_dataset(work / "synthetic_data"))
    cfg["data"]["synthetic"] = True
    cfg["paths"] = {"cache": str(work / "cache"), "results": str(work / "results")}
    cfg["cohorts"]["pilot"] = {"train": 4, "val": 2}
    cfg["cohorts"]["medium"] = {"train": 4, "val": None}
    cfg["model"].update(hidden=8, head_width=16, static_embedding=4, dropout=0.0)
    cfg["training"].update(
        epochs=2,
        pilot_epochs=2,
        full_epochs=2,
        batch_size=4,
        threads=1,
        allow_full_cpu=True,
        device="cpu",
        learning_rates=[0.0005],
    )
    cfg["evaluation"]["bootstrap_samples"] = 30
    return cfg
