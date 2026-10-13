"""Metrik kualitas citra: PSNR, SSIM, ΔE2000 (dengan mask opsional) dan metrik tanpa acuan (F4, F12).

Semua fungsi menerima citra RGB uint8 berbentuk (H, W, 3). Mask (opsional) adalah
array boolean (H, W); metrik hanya dihitung di piksel bernilai True.
"""

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

DATA_RANGE = 255.0


def _check(pred, ref, mask):
    if pred.shape != ref.shape:
        raise ValueError(f"ukuran berbeda: {pred.shape} vs {ref.shape}")
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != pred.shape[:2]:
            raise ValueError(f"ukuran mask {mask.shape} tidak cocok dengan {pred.shape[:2]}")
        if not mask.any():
            raise ValueError("mask kosong")
    return mask


def psnr(pred, ref, mask=None):
    """PSNR dalam dB dengan data_range 255, dihitung atas seluruh kanal RGB."""
    mask = _check(pred, ref, mask)
    err = (pred.astype(np.float64) - ref.astype(np.float64)) ** 2
    mse = err[mask].mean() if mask is not None else err.mean()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(DATA_RANGE ** 2 / mse))


def ssim(pred, ref, mask=None):
    """SSIM pada RGB (rata-rata ketiga kanal), jendela Gaussian σ=1.5 seperti Wang dkk. (2004).

    Dengan mask, peta SSIM dirata-rata hanya di dalam mask.
    """
    mask = _check(pred, ref, mask)
    score, smap = structural_similarity(
        pred, ref, channel_axis=2, data_range=DATA_RANGE,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    if mask is None:
        return float(score)
    return float(smap.mean(axis=2)[mask].mean())


def delta_e(pred, ref, mask=None):
    """Rata-rata ΔE CIEDE2000 di ruang CIELAB (illuminant D65). Makin kecil makin mirip."""
    mask = _check(pred, ref, mask)
    de = deltaE_ciede2000(rgb2lab(pred), rgb2lab(ref))
    return float(de[mask].mean() if mask is not None else de.mean())


def luminance(rgb):
    """Luma BT.601 (0-255) sebagai float."""
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def entropy(gray_u8):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def no_reference(rgb):
    """Metrik tanpa acuan: rata-rata luminans, simpangan baku luminans, entropi (bit)."""
    y = luminance(rgb)
    return {"mean_lum": float(y.mean()), "std_lum": float(y.std()),
            "entropy": entropy(np.clip(np.rint(y), 0, 255).astype(np.uint8))}


def full_reference(pred, ref, mask=None):
    return {"psnr": psnr(pred, ref, mask), "ssim": ssim(pred, ref, mask),
            "delta_e": delta_e(pred, ref, mask)}
