"""OpenCV display helpers for large images."""

import cv2
import numpy as np

from .config import MAX_DISPLAY_HEIGHT, MAX_DISPLAY_WIDTH


def fit_to_window(
    image: np.ndarray,
    max_width: int = MAX_DISPLAY_WIDTH,
    max_height: int = MAX_DISPLAY_HEIGHT,
) -> np.ndarray:
    """Scale an image down to fit the display window without upscaling."""
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return image

    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def show_image(image: np.ndarray, window_name: str = "Image", wait: bool = True) -> None:
    """Show an image in an OpenCV window, optionally waiting for a key press."""
    cv2.imshow(window_name, fit_to_window(image))
    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def show_pair(
    original: np.ndarray,
    result: np.ndarray,
    window_name: str = "Before / After",
    wait: bool = True,
) -> None:
    """Show two images side by side, padded to the same height."""
    height = max(original.shape[0], result.shape[0])
    padded = []
    for image in (original, result):
        missing = height - image.shape[0]
        padded.append(
            cv2.copyMakeBorder(
                image,
                0,
                missing,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )

    show_image(np.hstack(padded), window_name=window_name, wait=wait)
