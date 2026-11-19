"""Shared fixtures and helpers for the test suite.

Run everything from the project root with:  python -m pytest
"""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_gui(monkeypatch):
    """Never open an OpenCV window during tests (keeps the suite headless)."""
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: -1)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda *a, **k: -1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)


def rand_img(h: int, w: int, seed: int = 0, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (h, w, channels) if channels else (h, w)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


@pytest.fixture
def img():
    """Small random BGR image (height 10, width 12)."""
    return rand_img(10, 12, seed=42)


@pytest.fixture
def write_png(tmp_path):
    """Write an image to a temp PNG and return its path."""

    def _write(image: np.ndarray, name: str = "input.png") -> Path:
        path = tmp_path / name
        assert cv2.imwrite(str(path), image)
        return path

    return _write


def assert_valid_vertical_seam(seam, height: int, width: int) -> None:
    seam = np.asarray(seam)
    assert seam.shape == (height,)
    assert seam.min() >= 0 and seam.max() < width
    assert np.all(np.abs(np.diff(seam.astype(int))) <= 1)


def best_seam_cost(energy: np.ndarray) -> float:
    """Brute-force optimum over every connected vertical seam (tiny inputs only)."""
    h, w = energy.shape

    def walk(r, c):
        if r == h - 1:
            return float(energy[r, c])
        return float(energy[r, c]) + min(walk(r + 1, j) for j in range(max(c - 1, 0), min(c + 1, w - 1) + 1))

    return min(walk(0, c) for c in range(w))


def is_row_subsequence(original_row: np.ndarray, carved_row: np.ndarray) -> bool:
    """True if carved_row is original_row with some pixels deleted (order preserved)."""
    i = 0
    for px in original_row:
        if i < len(carved_row) and np.array_equal(px, carved_row[i]):
            i += 1
    return i == len(carved_row)
