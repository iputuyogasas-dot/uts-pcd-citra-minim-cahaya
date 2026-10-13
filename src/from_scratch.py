"""Dua operasi dari nol (PRD bagian 8): ekualisasi histogram dan filter median.

Hanya memakai NumPy. Perilaku dibuat sama dengan pembandingnya di OpenCV
(cv2.equalizeHist dan cv2.medianBlur) agar selisih piksel bisa diverifikasi.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _as_2d_uint8(img, name):
    """Terima citra (H, W) atau (H, W, 1) uint8; kembalikan versi 2D dan bentuk aslinya."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise TypeError(f"{name}: citra harus uint8, bukan {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0], img.shape
    if img.ndim != 2:
        raise ValueError(f"{name}: citra harus satu kanal (H, W), bukan {img.shape}")
    return img, img.shape


def histogram(img):
    """Histogram 256 bin dari citra uint8."""
    return np.bincount(img.ravel(), minlength=256)


def equalize_hist(img):
    """Ekualisasi histogram citra satu kanal uint8, setara cv2.equalizeHist.

    Langkah:
      1. Hitung histogram 256 bin dan CDF-nya.
      2. Cari bin pertama yang tidak kosong, i (cdf_min = hist[i]).
      3. LUT[j] = round((cdf[j] - cdf_min) * 255 / (N - cdf_min)), dengan LUT[i] = 0.
      4. Petakan setiap piksel lewat LUT.

    Detail yang disamakan dengan OpenCV:
      - Citra konstan (semua piksel di satu bin) dikembalikan apa adanya.
      - Skala dihitung dalam float32, dan pembulatan memakai round-half-to-even
        (np.rint), sama dengan cvRound.
    """
    gray, shape = _as_2d_uint8(img, "equalize_hist")
    hist = histogram(gray)
    total = gray.size

    i = int(np.flatnonzero(hist)[0])
    if hist[i] == total:
        return gray.copy().reshape(shape)

    scale = np.float32(255.0) / np.float32(total - hist[i])
    cdf = np.cumsum(hist)
    lut = np.rint((cdf - cdf[i]).astype(np.float32) * scale)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    lut[:i] = 0  # bin kosong sebelum i tidak pernah dipakai

    return lut[gray].reshape(shape)


def _median_2d(channel, ksize, max_bytes):
    """Median satu kanal dengan padding replikasi tepi, diproses per potongan baris."""
    r = ksize // 2
    padded = np.pad(channel, r, mode="edge")
    h, w = channel.shape
    mid = ksize * ksize // 2

    # Setiap baris keluaran butuh w * k * k elemen sementara; batasi agar memori aman.
    row_bytes = w * ksize * ksize * channel.itemsize
    chunk = max(1, max_bytes // row_bytes)

    out = np.empty_like(channel)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        windows = sliding_window_view(padded[y0:y1 + 2 * r], (ksize, ksize))
        flat = windows.reshape(y1 - y0, w, ksize * ksize)
        # k*k selalu ganjil, jadi median = elemen ke-mid setelah diurutkan.
        out[y0:y1] = np.partition(flat, mid, axis=-1)[..., mid]
    return out


def median_filter(img, ksize=3, max_bytes=64 * 1024 * 1024):
    """Filter median, setara cv2.medianBlur.

    Args:
        img: citra uint8 berbentuk (H, W) atau (H, W, C); tiap kanal diproses terpisah.
        ksize: ukuran kernel ganjil >= 1.
        max_bytes: batas memori sementara per potongan baris.

    Tepi ditangani dengan replikasi piksel tepi (np.pad mode='edge'),
    sama dengan BORDER_REPLICATE yang dipakai cv2.medianBlur.
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise TypeError(f"median_filter: citra harus uint8, bukan {img.dtype}")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"median_filter: ksize harus ganjil >= 1, bukan {ksize}")
    if img.ndim not in (2, 3):
        raise ValueError(f"median_filter: bentuk citra tidak didukung {img.shape}")
    if ksize == 1:
        return img.copy()

    if img.ndim == 2:
        return _median_2d(img, ksize, max_bytes)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _median_2d(np.ascontiguousarray(img[:, :, c]), ksize, max_bytes)
    return out
