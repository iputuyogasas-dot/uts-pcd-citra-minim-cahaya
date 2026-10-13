"""Verifikasi operasi dari nol terhadap OpenCV (PRD bagian 8, T19)."""

import cv2
import numpy as np
import pytest

from src.from_scratch import equalize_hist, histogram, median_filter

rng = np.random.default_rng(42)

SIZES = [(1, 1), (1, 17), (17, 1), (32, 32), (31, 47), (64, 50), (101, 99)]


def _random(shape, low=0, high=256):
    return rng.integers(low, high, size=shape, dtype=np.uint8)


# ---------- ekualisasi histogram ----------

def test_histogram_matches_cv2():
    img = _random((40, 60))
    ref = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(int)
    assert np.array_equal(histogram(img), ref)


@pytest.mark.parametrize("shape", SIZES)
def test_equalize_random(shape):
    img = _random(shape)
    assert np.array_equal(equalize_hist(img), cv2.equalizeHist(img))


@pytest.mark.parametrize("low,high", [(0, 20), (10, 40), (200, 256), (0, 2)])
def test_equalize_narrow_range(low, high):
    # Rentang sempit meniru citra gelap/minim kontras.
    img = _random((80, 120), low, high)
    assert np.array_equal(equalize_hist(img), cv2.equalizeHist(img))


@pytest.mark.parametrize("value", [0, 1, 128, 255])
def test_equalize_constant(value):
    img = np.full((20, 30), value, np.uint8)
    out = equalize_hist(img)
    assert np.array_equal(out, cv2.equalizeHist(img))
    assert np.array_equal(out, img)


def test_equalize_single_channel_3d():
    img = _random((25, 35, 1))
    out = equalize_hist(img)
    assert out.shape == img.shape
    assert np.array_equal(out[:, :, 0], cv2.equalizeHist(img[:, :, 0]))


def test_equalize_rejects_color_and_float():
    with pytest.raises(ValueError):
        equalize_hist(_random((10, 10, 3)))
    with pytest.raises(TypeError):
        equalize_hist(np.zeros((10, 10), np.float32))


# ---------- filter median ----------

@pytest.mark.parametrize("ksize", [3, 5, 7, 9])
@pytest.mark.parametrize("shape", SIZES)
def test_median_gray(shape, ksize):
    img = _random(shape)
    assert np.array_equal(median_filter(img, ksize), cv2.medianBlur(img, ksize))


@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_median_color(ksize):
    img = _random((45, 64, 3))
    assert np.array_equal(median_filter(img, ksize), cv2.medianBlur(img, ksize))


@pytest.mark.parametrize("ksize", [3, 5])
def test_median_constant(ksize):
    img = np.full((15, 15), 77, np.uint8)
    assert np.array_equal(median_filter(img, ksize), cv2.medianBlur(img, ksize))


def test_median_chunking_does_not_change_result():
    img = _random((50, 40))
    full = median_filter(img, 5)
    tiny = median_filter(img, 5, max_bytes=1)  # memaksa satu baris per potongan
    assert np.array_equal(full, tiny)


def test_median_ksize_1_is_identity():
    img = _random((10, 10))
    assert np.array_equal(median_filter(img, 1), img)


@pytest.mark.parametrize("ksize", [0, 2, 4, -3])
def test_median_rejects_bad_ksize(ksize):
    with pytest.raises(ValueError):
        median_filter(_random((10, 10)), ksize)
