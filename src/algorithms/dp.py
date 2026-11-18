"""Dynamic programming seam carving (Member 1: DP core engine)."""

from typing import Any, Callable

import cv2
import numpy as np

from ..display import fit_to_window
from ..utils import draw_seam, draw_seams, remove_vertical_seam, sobel_edge_map
from ..utils.filters import SOBEL_BORDER
from ..utils.timing import TimerResult, timer
from .both_directions import carve_both_directions
from .base import AlgorithmResult, SeamCarver


def _compute_energy(image: np.ndarray) -> np.ndarray:
    return sobel_edge_map(image).astype(np.float32)


def _update_energy(image: np.ndarray, energy: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Recalculate Sobel energy around affected pixels after seam removal."""
    h, w = image.shape[:2]
    lo = max(0, int(seam.min()) - 2)
    hi = min(w, int(seam.max()) + 2)
    if lo >= hi:
        return energy

    pad_lo = max(0, lo - 1)
    pad_hi = min(w, hi + 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    region = gray[:, pad_lo:pad_hi]
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3, borderType=SOBEL_BORDER)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3, borderType=SOBEL_BORDER)
    region_energy = np.abs(gx) + np.abs(gy)
    energy[:, lo:hi] = region_energy[:, lo - pad_lo : lo - pad_lo + (hi - lo)]
    return energy


def _find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Find minimum-energy vertical seam using DP and backtracking."""
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        left = np.roll(dp[i - 1], 1)
        left[0] = np.inf
        right = np.roll(dp[i - 1], -1)
        right[-1] = np.inf
        middle = dp[i - 1]
        choices = np.vstack((left, middle, right))
        idx = np.argmin(choices, axis=0) - 1  # -1,0,1 offset
        backtrack[i] = np.arange(w) + idx
        dp[i] += choices[idx + 1, np.arange(w)]

    seam = np.empty(h, dtype=np.int32)
    seam[h - 1] = int(np.argmin(dp[h - 1]))
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    return seam


def _transpose(image: np.ndarray) -> np.ndarray:
    """Swap rows and columns: (H, W[, C]) -> (W, H[, C]).

    A horizontal seam of an image is a vertical seam of its transpose, so
    horizontal carving reuses the vertical code on the transposed image.
    """
    # .copy() (not a view) gives a normal C-ordered array that OpenCV accepts
    return image.swapaxes(0, 1).copy()


def _show_step(frame: np.ndarray, title: str, delay: int = 1) -> int:
    cv2.imshow(title, fit_to_window(frame))
    return cv2.waitKey(delay) & 0xFF


class DPSeamCarver(SeamCarver):
    """Optimal vertical or horizontal seam removal using dynamic programming."""

    name = "dp"
    description = "Dynamic programming optimal seam carving"

    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        """Remove ``num_seams`` seams.

        ``direction="vertical"`` (default) removes columns (narrower image).
        ``direction="horizontal"`` removes rows (shorter image): the image is
        transposed, carved with the same vertical DP, and transposed back.
        ``direction="both"`` removes ``num_seams`` vertical seams, then
        ``num_seams`` horizontal seams (see both_directions.py).
        For horizontal seams, ``seam[c]`` in the metadata is the row removed
        from column ``c``.
        """
        direction: str = kwargs.get("direction", "vertical")
        if direction == "both":
            # vertical seams first, then horizontal (see both_directions.py)
            return carve_both_directions(self, image, num_seams, **kwargs)
        if direction not in ("vertical", "horizontal"):
            raise ValueError("direction must be 'vertical', 'horizontal' or 'both'")
        horizontal = direction == "horizontal"

        if num_seams < 0:
            raise ValueError("num_seams must be >= 0")
        if horizontal and num_seams >= image.shape[0]:
            raise ValueError("cannot remove more seams than image height")
        if not horizontal and num_seams >= image.shape[1]:
            raise ValueError("cannot remove more seams than image width")

        # Work on the transposed image for horizontal seams; `to_view` turns
        # anything we show or return back into the normal orientation.
        work = _transpose(image) if horizontal else image
        to_view = _transpose if horizontal else (lambda img: img)

        visualize: bool = kwargs.get("visualize", False)
        step_delay: int = int(kwargs.get("step_delay", 1))
        show_all_seams: bool = kwargs.get("show_all_seams", visualize)
        window_name: str = kwargs.get("window_name", "Seam Carving - Step by Step")
        step_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("step_callback")
        keep_steps: bool = kwargs.get("keep_steps", False)
        # Step frames are only drawn when something will use them. Drawing (and
        # keeping) a full-size frame for every seam costs time and a lot of memory.
        need_frames = visualize or step_callback is not None or keep_steps

        result_image = work.copy()
        original = work.copy()
        h, w = work.shape[:2]
        col_map = np.broadcast_to(np.arange(w), (h, w)).copy()
        seams: list[np.ndarray] = []
        seams_original: list[np.ndarray] = []
        step_frames: list[np.ndarray] = []
        elapsed = TimerResult()

        with timer(elapsed):
            energy = _compute_energy(result_image)
            for i in range(num_seams):
                seam = _find_vertical_seam(energy)
                seams.append(seam.copy())
                seam_orig = col_map[np.arange(h), seam].copy()
                seams_original.append(seam_orig)

                if need_frames:
                    frame = to_view(draw_seam(result_image, seam, color=(0, 0, 255)))
                    cv2.putText(
                        frame,
                        f"Seam {i + 1}/{num_seams}  size: {frame.shape[1]}x{frame.shape[0]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if keep_steps:
                        step_frames.append(frame.copy())

                    if step_callback:
                        step_callback(
                            {
                                "step": i + 1,
                                "total": num_seams,
                                "seam": seam.copy(),
                                "seam_original": seam_orig,
                                "frame": frame,
                                "image": to_view(result_image.copy()),
                            }
                        )
                    if visualize:
                        _show_step(frame, window_name, delay=step_delay)

                result_image = remove_vertical_seam(result_image, seam)
                # Delete the seam's pixel from every row of the energy map and the
                # column map in one step (instead of shifting each row with np.roll).
                keep = np.ones(energy.shape, dtype=bool)
                keep[np.arange(h), seam] = False
                energy = energy[keep].reshape(h, -1)
                col_map = col_map[keep].reshape(h, -1)
                if energy.shape[1] > 0:
                    energy = _update_energy(result_image, energy, seam)

        all_seams_image = (
            to_view(draw_seams(original, seams_original, color=(0, 0, 255))) if show_all_seams else None
        )
        if visualize and all_seams_image is not None:
            cv2.imshow("All Seams (on original)", fit_to_window(all_seams_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return AlgorithmResult(
            image=to_view(result_image),
            algorithm=self.name,
            runtime_s=elapsed.elapsed_s,
            metadata={
                "direction": direction,
                "seams_removed": num_seams,
                "seams": seams,
                "seams_original": seams_original,
                "all_seams_image": all_seams_image,
                "step_frames": step_frames,
            },
        )
