import cv2
import numpy as np


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Remove a single vertical seam from an image.

    `seam` must have one column index per row, where `seam[i]` is removed from row `i`.
    """
    h, w = image.shape[:2]
    seam = np.asarray(seam)
    if seam.shape[0] != h:
        raise ValueError("seam length must match image height")

    if image.ndim == 3:
        out = np.empty((h, w - 1, image.shape[2]), dtype=image.dtype)
    else:
        out = np.empty((h, w - 1), dtype=image.dtype)

    for i in range(h):
        j = int(seam[i])
        if j < 0 or j >= w:
            raise ValueError("seam index out of bounds")
        out[i] = np.delete(image[i], j, axis=0)

    return out


def draw_seam(image: np.ndarray, seam: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Return a copy of image with a single vertical seam drawn.

    A 2-D (grayscale) image is converted to BGR first so the colored seam is visible.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for i, j in enumerate(np.asarray(seam)):
        j = int(j)
        if 0 <= i < vis.shape[0] and 0 <= j < vis.shape[1]:
            vis[i, j] = color
    return vis


def draw_seams(image: np.ndarray, seams: list[np.ndarray], color=(0, 0, 255)) -> np.ndarray:
    """Return a copy of image with multiple vertical seams drawn."""
    vis = image.copy()
    for seam in seams:
        vis = draw_seam(vis, seam, color=color)
    return vis
