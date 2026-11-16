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
