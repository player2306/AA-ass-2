"""Seam carving in both directions: all vertical seams first, then all horizontal.

Used when an algorithm's ``run()`` is called with ``direction="both"`` (for
example by the GUI's Remove button). The same algorithm (DP or greedy) finds
every seam, in two batches:

    1. remove ``num_seams`` vertical seams    -> image becomes narrower
    2. remove ``num_seams`` horizontal seams  -> image becomes shorter

Each batch is one normal ``run()`` call, so the algorithm works exactly as it
does for a single direction (DP keeps its incremental energy update), and the
energy map is always recalculated after each removed seam.
"""

from typing import Any, Callable

import cv2
import numpy as np

from ..display import fit_to_window
from .base import AlgorithmResult

RED = (0, 0, 255)  # BGR

# Options that this file handles itself (drawing / showing steps), so they are
# not passed on to the two batch runs.
_DISPLAY_OPTIONS = {"direction", "step_callback", "visualize", "step_delay",
                    "show_all_seams", "window_name", "keep_steps"}


def _seam_pixels(seam: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the pixels a seam removes.

    vertical:   seam[r] is the column removed from row r
    horizontal: seam[c] is the row removed from column c
    """
    seam = np.asarray(seam)
    steps = np.arange(len(seam))
    return (steps, seam) if direction == "vertical" else (seam, steps)


def _delete(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, direction: str) -> np.ndarray:
    """Remove the seam's pixels from a 2-D array (one per row, or one per column)."""
    keep = np.ones(grid.shape, dtype=bool)
    keep[rows, cols] = False
    h, w = grid.shape
    if direction == "vertical":
        return grid[keep].reshape(h, w - 1)
    # horizontal: work column by column via the transpose
    return grid.T[keep.T].reshape(w, h - 1).T


def carve_both_directions(carver, image: np.ndarray, num_seams: int, **kwargs: Any) -> AlgorithmResult:
    """Remove ``num_seams`` vertical seams, then ``num_seams`` horizontal seams.

    Accepts the same display options as the algorithms (step_callback,
    visualize, step_delay, show_all_seams, window_name). Any other options
    (e.g. multi_start for greedy) are passed on to ``carver.run``.
    """
    height, width = image.shape[:2]
    if num_seams < 0:
        raise ValueError("num_seams must be >= 0")
    if num_seams >= width or num_seams >= height:
        raise ValueError("cannot remove more seams than image width or height")

    step_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("step_callback")
    visualize: bool = kwargs.get("visualize", False)
    step_delay: int = int(kwargs.get("step_delay", 1))
    show_all_seams: bool = kwargs.get("show_all_seams", visualize)
    window_name: str = kwargs.get("window_name", "Seam Carving - Step by Step")
    options = {k: v for k, v in kwargs.items() if k not in _DISPLAY_OPTIONS}
    total = 2 * num_seams

    def relay(direction: str, offset: int):
        """Step callback for one batch: redraws the frame with the overall step
        number (e.g. "Seam 7/10 (horizontal)") and passes it on."""
        def on_step(info: dict[str, Any]) -> None:
            step = offset + info["step"]
            current = info["image"]  # current image, normal orientation
            frame = current.copy()
            rows, cols = _seam_pixels(info["seam"], direction)
            frame[rows, cols] = RED
            cv2.putText(
                frame,
                f"Seam {step}/{total} ({direction})  size: {frame.shape[1]}x{frame.shape[0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if step_callback:
                step_callback({"step": step, "total": total, "direction": direction,
                               "seam": info["seam"], "frame": frame, "image": current})
            if visualize:
                cv2.imshow(window_name, fit_to_window(frame))
                cv2.waitKey(step_delay)
        return on_step

    # Frames are only drawn if someone will look at them.
    show_steps = step_callback is not None or visualize

    # Batch 1: vertical seams, batch 2: horizontal seams on the narrower image
    first = carver.run(image, num_seams=num_seams, direction="vertical",
                       step_callback=relay("vertical", 0) if show_steps else None, **options)
    second = carver.run(first.image, num_seams=num_seams, direction="horizontal",
                        step_callback=relay("horizontal", num_seams) if show_steps else None, **options)

    seams = [("vertical", s) for s in first.metadata["seams"]] + \
            [("horizontal", s) for s in second.metadata["seams"]]

    all_seams_image = None
    if show_all_seams:
        # Follow every pixel back to its place in the ORIGINAL image: remove each
        # seam (in order) from two maps that hold the original row / column.
        orig_rows = np.repeat(np.arange(height)[:, None], width, axis=1)
        orig_cols = np.tile(np.arange(width), (height, 1))
        removed = np.zeros((height, width), dtype=bool)
        for direction, seam in seams:
            rows, cols = _seam_pixels(seam, direction)
            removed[orig_rows[rows, cols], orig_cols[rows, cols]] = True
            orig_rows = _delete(orig_rows, rows, cols, direction)
            orig_cols = _delete(orig_cols, rows, cols, direction)
        all_seams_image = image.copy()
        all_seams_image[removed] = RED
    if visualize and all_seams_image is not None:
        cv2.imshow("All Seams (on original)", fit_to_window(all_seams_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    metadata: dict[str, Any] = {
        "direction": "both",
        "seams_removed": total,
        "seams": seams,  # list of (direction, seam) in removal order
        "all_seams_image": all_seams_image,
    }
    energies = first.metadata.get("seam_energies", []) + second.metadata.get("seam_energies", [])
    if len(energies) == total:  # only if the algorithm reports them (greedy)
        metadata["seam_energies"] = energies

    return AlgorithmResult(
        image=second.image,
        algorithm=second.algorithm,
        runtime_s=first.runtime_s + second.runtime_s,  # algorithm time only
        metadata=metadata,
    )
