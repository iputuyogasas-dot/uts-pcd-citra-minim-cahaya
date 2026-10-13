"""Uji Wilcoxon signed-rank untuk membandingkan dua metode pada data berpasangan (F10)."""

import pandas as pd
from scipy.stats import wilcoxon

HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "delta_e": False}


def wilcoxon_compare(df, method_a, method_b, metric):
    """Bandingkan A vs B pada citra yang sama. Kembalikan median selisih (A-B), p-value, dan pemenang."""
    piv = df[df.method.isin([method_a, method_b])].pivot_table(
        index="image", columns="method", values=metric)
    piv = piv.dropna()
    diff = piv[method_a] - piv[method_b]
    if (diff == 0).all():
        p = 1.0
    else:
        p = float(wilcoxon(piv[method_a], piv[method_b], zero_method="wilcox").pvalue)
    better_a = diff.median() > 0 if HIGHER_IS_BETTER[metric] else diff.median() < 0
    return {"a": method_a, "b": method_b, "metric": metric, "n": len(piv),
            "median_diff": float(diff.median()), "mean_diff": float(diff.mean()),
            "a_wins": int((diff > 0).sum() if HIGHER_IS_BETTER[metric] else (diff < 0).sum()),
            "p_value": p, "better": "seri" if (diff == 0).all() else (method_a if better_a else method_b),
            "significant_0.05": p < 0.05}


def compare_many(df, pairs, metrics=("psnr", "ssim", "delta_e")):
    return pd.DataFrame([wilcoxon_compare(df, a, b, m) for a, b in pairs for m in metrics])
