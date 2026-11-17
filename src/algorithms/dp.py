"""Dynamic programming seam carving (Member 1: DP core engine)."""

from typing import Any, Callable

import cv2
import numpy as np

from ..display import fit_to_window
from ..utils import draw_seam, draw_seams, remove_vertical_seam, sobel_edge_map
from ..utils.timing import TimerResult, timer
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
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
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


def _show_step(frame: np.ndarray, title: str, delay: int = 1) -> int:
    cv2.imshow(title, fit_to_window(frame))
    return cv2.waitKey(delay) & 0xFF


class DPSeamCarver(SeamCarver):
    """Optimal vertical seam removal using dynamic programming."""

    name = "dp"
    description = "Dynamic programming optimal seam carving"

    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        if num_seams < 0:
            raise ValueError("num_seams must be >= 0")
        if num_seams >= image.shape[1]:
            raise ValueError("cannot remove more seams than image width")

        visualize: bool = kwargs.get("visualize", False)
        step_delay: int = int(kwargs.get("step_delay", 1))
        show_all_seams: bool = kwargs.get("show_all_seams", visualize)
        window_name: str = kwargs.get("window_name", "Seam Carving - Step by Step")
        step_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("step_callback")

        result_image = image.copy()
        original = image.copy()
        h, w = image.shape[:2]
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

                frame = draw_seam(result_image, seam, color=(0, 0, 255))
                cv2.putText(
                    frame,
                    f"Seam {i + 1}/{num_seams}  size: {result_image.shape[1]}x{result_image.shape[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                step_frames.append(frame.copy())

                if step_callback:
                    step_callback(
                        {
                            "step": i + 1,
                            "total": num_seams,
                            "seam": seam.copy(),
                            "seam_original": seam_orig,
                            "frame": frame,
                            "image": result_image.copy(),
                        }
                    )
                if visualize:
                    _show_step(frame, window_name, delay=step_delay)

                result_image = remove_vertical_seam(result_image, seam)
                for r, j in enumerate(seam):
                    energy[r, j:] = np.roll(energy[r, j:], -1)
                    col_map[r, j:] = np.roll(col_map[r, j:], -1)
                energy = energy[:, :-1]
                col_map = col_map[:, :-1]
                if energy.shape[1] > 0:
                    energy = _update_energy(result_image, energy, seam)

        all_seams_image = draw_seams(original, seams_original, color=(0, 0, 255)) if show_all_seams else None
        if visualize and all_seams_image is not None:
            cv2.imshow("All Seams (on original)", fit_to_window(all_seams_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return AlgorithmResult(
            image=result_image,
            algorithm=self.name,
            runtime_s=elapsed.elapsed_s,
            metadata={
                "seams_removed": num_seams,
                "seams": seams,
                "seams_original": seams_original,
                "all_seams_image": all_seams_image,
                "step_frames": step_frames if kwargs.get("keep_steps", False) else [],
            },
        )
