"""Membaca dataset, memetakan kondisi, dan memvalidasi kelengkapan pasangan (F1)."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src import config

NAME_RE = re.compile(r"^(s\d{2})_(acuan|k[1-5])\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class Pair:
    scenario: str      # "s01"
    condition: str     # "K1".."K5"
    dark_path: Path
    ref_path: Path

    @property
    def name(self):
        return f"{self.scenario}_{self.condition.lower()}"


def scan(data_dir=None):
    """Kelompokkan file per skenario: {"s01": {"acuan": path, "k1": path, ...}}."""
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {data_dir}")
    groups = {}
    for p in sorted(data_dir.iterdir()):
        m = NAME_RE.match(p.name)
        if m:
            groups.setdefault(m.group(1).lower(), {})[m.group(2).lower()] = p
    return groups


def list_pairs(data_dir=None, scenarios=None, validate=True):
    """Kembalikan daftar Pair (gelap, acuan), diurutkan per skenario lalu kondisi."""
    groups = scan(data_dir)
    if validate:
        validate_groups(groups)
    pairs = []
    for sc in sorted(groups):
        if scenarios is not None and sc not in scenarios:
            continue
        files = groups[sc]
        for k in range(1, 6):
            key = f"k{k}"
            if key in files and "acuan" in files:
                pairs.append(Pair(sc, f"K{k}", files[key], files["acuan"]))
    return pairs


def validate_groups(groups, expected_scenarios=10):
    """Pastikan tiap skenario punya 1 acuan + 5 gelap dan semua ukuran sama."""
    problems = []
    if len(groups) != expected_scenarios:
        problems.append(f"jumlah skenario {len(groups)}, seharusnya {expected_scenarios}")
    needed = {"acuan", "k1", "k2", "k3", "k4", "k5"}
    sizes = {}
    for sc, files in groups.items():
        missing = needed - set(files)
        if missing:
            problems.append(f"{sc}: file hilang {sorted(missing)}")
        for key, p in files.items():
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
            if img is None:
                problems.append(f"{p.name}: tidak bisa dibaca")
            else:
                sizes[p.name] = img.shape
    if len(set(sizes.values())) > 1:
        problems.append(f"ukuran citra berbeda: {sorted(set(sizes.values()))}")
    if problems:
        raise ValueError("Dataset tidak valid:\n  " + "\n  ".join(problems))
    return True


@lru_cache(maxsize=16)
def load_image(path, scale=config.WORK_SCALE):
    """Baca citra sebagai RGB uint8, diperkecil dengan INTER_AREA bila scale < 1.

    Hasil perkecilan disimpan sebagai PNG (lossless) di data/cache/ agar JPEG
    9.4 MP tidak perlu didekode ulang di setiap eksperimen.
    """
    path = Path(path)
    cache = config.CACHE_DIR / f"{path.stem}_x{scale:g}.png"
    bgr = cv2.imread(str(cache), cv2.IMREAD_COLOR) if scale != 1.0 and cache.exists() else None
    if bgr is None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Tidak bisa membaca {path}")
        if scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Tulis ke file sementara lalu rename: aman bila beberapa proses paralel menulis bersamaan.
            tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.png")
            cv2.imwrite(str(tmp), bgr)
            os.replace(tmp, cache)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.setflags(write=False)  # cache dipakai bersama; cegah perubahan tak sengaja
    return rgb


def load_pair(pair, scale=config.WORK_SCALE):
    return load_image(pair.dark_path, scale), load_image(pair.ref_path, scale)


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
