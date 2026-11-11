"""Image loading and saving helpers built on OpenCV."""

from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_IMAGE


def load_image(path: str | Path = DEFAULT_IMAGE) -> np.ndarray:
    """Load an image as a BGR uint8 array.

    Raises:
        FileNotFoundError: If the path does not exist or cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write an image to disk, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV could not write image: {path}")
    return path
