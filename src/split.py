"""Membuat dan membaca split dev/uji per skenario (F2).

Split dibuat sekali dengan seed tetap dan dikunci: skrip menolak menimpa
split.json yang sudah ada kecuali dipanggil dengan --force.
"""

import argparse
import json
from datetime import date

import numpy as np

from src import config
from src.dataset import scan


def make_split(scenarios, seed=config.SEED, n_dev=config.N_DEV_SCENARIOS):
    order = np.random.default_rng(seed).permutation(sorted(scenarios))
    dev, test = sorted(order[:n_dev].tolist()), sorted(order[n_dev:].tolist())
    return {"seed": seed, "unit": "scenario", "dev": dev, "test": test,
            "created": date.today().isoformat(), "locked": True}


def load_split(path=config.SPLIT_PATH):
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    if set(split["dev"]) & set(split["test"]):
        raise ValueError("split.json rusak: skenario muncul di dev dan uji")
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="timpa split.json yang sudah ada")
    args = ap.parse_args()

    if config.SPLIT_PATH.exists() and not args.force:
        print(f"{config.SPLIT_PATH} sudah ada dan terkunci; tidak diubah.")
        print(json.dumps(load_split(), indent=2))
        return
    split = make_split(scan().keys())
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
