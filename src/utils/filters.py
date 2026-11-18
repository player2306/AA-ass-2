import cv2
import numpy as np

# Border mode for every Sobel call in the project (shared so the full energy map
# and DP's incremental energy update always agree).
SOBEL_BORDER = cv2.BORDER_REPLICATE


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Sobel gradients as (gx, gy) using two passes."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=SOBEL_BORDER)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=SOBEL_BORDER)
    return gx, gy


def sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Compute a 2-pass Sobel edge magnitude map.

    Pass 1 computes horizontal gradients (Gx).
    Pass 2 computes vertical gradients (Gy).
    The result is the L1 magnitude |Gx| + |Gy|.
    """
    gx, gy = sobel_gradients(image)
    return np.abs(gx) + np.abs(gy)
