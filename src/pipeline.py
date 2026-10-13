"""Pipeline terkonfigurasi (F3).

Tahap:
  [1] pra-pemrosesan   : validasi + (opsional) white balance
  [2] reduksi derau    : pada RGB
  [3] perbaikan kontras: pada kanal luminans (atau tiap kanal RGB)
  [4] pasca-proses     : unsharp mask pada luminans, gabung kanal, kembali ke RGB
Urutan [2] dan [3] bisa ditukar lewat config["order"].

Contoh config (lihat configs/*.json):
{
  "color_space": "ycrcb",
  "order": "denoise_first",
  "white_balance": null,
  "denoise": {"method": "nlm", "h": 5},
  "contrast": [{"method": "adaptive_gamma", "target": 0.4}],
  "sharpen": {"sigma": 1.0, "amount": 0.3},
  "saturation": 1.0
}

CLI:
  python -m src.pipeline input.jpg -o out.jpg -c configs/final.json
  python -m src.pipeline "DATASET NEW" -o results/enhanced -c configs/final.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src import config as cfg_mod
from src.enhance import CONTRAST, DENOISE, gray_world, saturation_boost, unsharp
from src.preprocess import apply_on_luminance, validate

DEFAULT = {
    "color_space": "ycrcb",
    "order": "denoise_first",
    "white_balance": None,
    "denoise": {"method": "none"},
    "contrast": [],
    "sharpen": None,
    "saturation": 1.0,
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return {**DEFAULT, **json.load(f)}


def _params(step):
    return {k: v for k, v in step.items() if k != "method"}


class Pipeline:
    def __init__(self, config=None):
        self.config = {**DEFAULT, **(config or {})}

    def _denoise(self, rgb):
        step = self.config["denoise"] or {"method": "none"}
        return DENOISE[step["method"]](rgb, **_params(step))

    def _contrast(self, rgb):
        steps = self.config["contrast"] or []
        if not steps:
            return np.array(rgb)

        def run(ch):
            for step in steps:
                ch = CONTRAST[step["method"]](ch, **_params(step))
            return ch
        return apply_on_luminance(rgb, run, self.config["color_space"])

    def _sharpen(self, rgb):
        s = self.config["sharpen"]
        if not s or s.get("amount", 0) == 0:
            return rgb
        # Pemertajaman selalu pada luminans agar tidak memunculkan tepi berwarna.
        space = "ycrcb" if self.config["color_space"] in ("rgb", "ratio") else self.config["color_space"]
        return apply_on_luminance(rgb, lambda ch: unsharp(ch, **s), space)

    def run(self, rgb, keep_intermediate=False):
        """Proses satu citra RGB uint8. Bila keep_intermediate, kembalikan juga dict keluaran tiap tahap."""
        c = self.config
        stages = {}
        x = validate(rgb)
        if c["white_balance"] == "gray_world":
            x = gray_world(x)
        stages["1_preprocess"] = x

        order = ["denoise", "contrast"] if c["order"] == "denoise_first" else ["contrast", "denoise"]
        for i, name in enumerate(order, start=2):
            x = self._denoise(x) if name == "denoise" else self._contrast(x)
            stages[f"{i}_{name}"] = x

        x = self._sharpen(x)
        if c.get("saturation", 1.0) != 1.0:
            x = saturation_boost(x, c["saturation"])
        stages["4_postprocess"] = x
        return (x, stages) if keep_intermediate else x

    __call__ = run


def main():
    from src.dataset import load_image, save_rgb

    ap = argparse.ArgumentParser(description="Jalankan pipeline pada satu citra atau satu folder.")
    ap.add_argument("input", help="file .jpg atau folder")
    ap.add_argument("-o", "--output", required=True, help="file keluaran atau folder keluaran")
    ap.add_argument("-c", "--config", default=str(cfg_mod.CONFIGS_DIR / "final.json"))
    ap.add_argument("--scale", type=float, default=cfg_mod.WORK_SCALE)
    args = ap.parse_args()

    pipe = Pipeline(load_config(args.config))
    src = Path(args.input)
    files = sorted(src.glob("*_k[1-5].jpg")) if src.is_dir() else [src]
    for f in files:
        out = pipe(np.array(load_image(f, args.scale)))
        dst = Path(args.output) / f.name if src.is_dir() else Path(args.output)
        save_rgb(dst, out)
        print(f"{f.name} -> {dst}")


if __name__ == "__main__":
    main()
