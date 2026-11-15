"""Small helpers shared by the seam carving algorithms.

A vertical seam is stored as a 1-D int array ``seam`` of length H, where
``seam[r]`` is the column of the pixel removed from row ``r``.
"""

import cv2
import numpy as np


def compute_energy(image: np.ndarray) -> np.ndarray:
    """Energy map = |dI/dx| + |dI/dy| of the grayscale image (Sobel filter).

    High energy = edges/detail (keep). Low energy = flat areas (safe to remove).
    Returns a float64 array of shape (H, W).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # BORDER_REPLICATE: the default (reflect) border makes the gradient 0 on the
    # image edges, which would pull every seam to the left/right border.
    border = cv2.BORDER_REPLICATE
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=border)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=border)
    return np.abs(dx) + np.abs(dy)


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` that is one column narrower, without the seam."""
    height, width = image.shape[:2]
    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam] = False  # mark one pixel per row to drop
    return image[keep].reshape(height, width - 1, *image.shape[2:])


def seam_energy(energy: np.ndarray, seam: np.ndarray) -> float:
    """Total energy of a seam (sum of energy of its pixels). Lower = better seam."""
    return float(energy[np.arange(len(seam)), seam].sum())


def draw_seam(image: np.ndarray, seam: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Return a copy of ``image`` with the seam pixels painted (default red in BGR)."""
    out = image.copy()
    out[np.arange(len(seam)), seam] = color
    return out
