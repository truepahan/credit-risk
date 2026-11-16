"""Разбиение исходного набора на исходную часть и порции «новых» данных.

В наборе нет даты заявки, поэтому разбиение случайное с фиксированным seed.
Исходный файл data/raw/credit_risk_dataset.csv не изменяется.

Запуск:  python -m src.data.split_data
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_params


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split(df: pd.DataFrame, initial_share: float, n_batches: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_initial = int(round(len(df) * initial_share))
    initial = df.iloc[idx[:n_initial]]
    batches = [df.iloc[part] for part in np.array_split(idx[n_initial:], n_batches)]
    return initial, batches


def main() -> None:
    p = load_params()
    raw = ROOT / p["data"]["raw_file"]
    df = pd.read_csv(raw)

    initial, batches = split(df, p["split"]["initial_share"], p["split"]["n_batches"], p["seed"])

    initial_path = ROOT / p["data"]["initial_file"]
    incoming = ROOT / p["data"]["incoming_dir"]
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    incoming.mkdir(parents=True, exist_ok=True)

    initial.to_csv(initial_path, index=False)
    files = [{"file": str(initial_path.relative_to(ROOT)), "rows": len(initial),
              "sha256": sha256(initial_path)}]
    for i, b in enumerate(batches, start=1):
        path = incoming / f"batch_{i:02d}.csv"
        b.to_csv(path, index=False)
        files.append({"file": str(path.relative_to(ROOT)), "rows": len(b), "sha256": sha256(path)})

    manifest = {
        "source": "Kaggle: Credit Risk Dataset (laotse), credit_risk_dataset.csv",
        "raw": {"file": str(raw.relative_to(ROOT)), "rows": len(df), "columns": df.shape[1],
                "sha256": sha256(raw)},
        "split": {"method": "random", "seed": p["seed"],
                  "initial_share": p["split"]["initial_share"],
                  "n_batches": p["split"]["n_batches"]},
        "files": files,
    }
    out = ROOT / "data" / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in files:
        print(f"{f['file']}: {f['rows']} строк")


if __name__ == "__main__":
    main()
