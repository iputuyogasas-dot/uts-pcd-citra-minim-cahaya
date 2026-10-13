"""Gaya grafik bersama untuk semua gambar laporan (Matplotlib, PNG statis)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Urutan kategori tetap; warna mengikuti entitas, bukan peringkat.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONDITION_COLORS = dict(zip(["K1", "K2", "K3", "K4", "K5"], CATEGORICAL))
HIGHLIGHT = CATEGORICAL[0]
BASE = "#c3c2b7"  # batang pembanding yang tidak disorot

# Sekuensial satu rona (biru), terang -> gelap.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "font.family": "DejaVu Sans",
    })


def heatmap(ax, values, row_labels, col_labels, fmt="{:.3f}", best=None, cmap=SEQUENTIAL):
    """Heatmap dengan nilai di tiap sel; `best` = (i, j) sel terpilih diberi bingkai."""
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)), [str(c) for c in col_labels])
    ax.set_yticks(range(len(row_labels)), [str(r) for r in row_labels])
    ax.grid(False)
    lo, hi = float(values.min()), float(values.max())
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = (v - lo) / (hi - lo + 1e-12) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK)
    if best is not None:
        i, j = best
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=CATEGORICAL[1], lw=2.2))
    return im
