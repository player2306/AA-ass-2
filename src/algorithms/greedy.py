"""Greedy seam carving (Member 3: greedy implementation)."""

import time
from typing import Any, Callable

import cv2
import numpy as np

from ..display import fit_to_window
from ..seam_utils import seam_energy
from ..utils import draw_seam, draw_seams, remove_vertical_seam, sobel_edge_map  # shared with DP
from ..utils.timing import TimerResult, timer
from .base import AlgorithmResult, SeamCarver


def find_greedy_seam(energy: np.ndarray, start: int | None = None) -> np.ndarray:
    """Find a vertical seam with a greedy, row-by-row local choice.

    1. Start at the lowest-energy pixel in the top row
       (or at column ``start`` if given; used by the tests for multi-start).
    2. For every next row, look only at the (up to) 3 pixels below the
       current one: down-left, down, down-right. Move to the cheapest.
    3. Never look further ahead and never undo a choice -> fast, but not
       guaranteed to find the optimal (lowest-energy) seam.

    Boundaries: the candidate window is clamped to [0, width-1], so a seam on
    the left/right edge only has 2 candidates (no wrap-around).
    Ties: np.argmin returns the first minimum, so the leftmost candidate wins.
    This makes the result deterministic.

    Time O(W + H) per seam: O(W) to scan the top row, then O(1) work for each
    of the H rows. Space O(H) for the returned seam, O(1) extra working space.
    """
    height, width = energy.shape
    seam = np.zeros(height, dtype=int)

    # cheapest pixel in the top row, unless a start column was given
    seam[0] = np.argmin(energy[0]) if start is None else start

    for r in range(1, height):
        prev = seam[r - 1]
        left = max(prev - 1, 0)
        right = min(prev + 1, width - 1)
        # argmin gives the position inside the slice, so add `left` back
        seam[r] = left + np.argmin(energy[r, left:right + 1])

    return seam


def find_multistart_greedy_seam(energy: np.ndarray) -> np.ndarray:
    """Multi-start greedy: run greedy from EVERY top-row column, keep the cheapest seam.

    This is an alternative design that fixes the "bad starting point" problem
    (one cheap top pixel leading into an expensive area). It is still greedy
    after the start, so it is still not guaranteed to find the optimal seam.

    All W greedy walks are done at the same time with NumPy: ``cols`` holds the
    current column of every walk, and each row moves all of them one step.
    Each step uses the same rules as find_greedy_seam (stay inside the image,
    ties go to the leftmost candidate), so walk j gives the same seam as
    find_greedy_seam(energy, start=j).

    Time O(H * W) per seam (same order as DP), extra space O(H * W) for the paths.
    """
    height, width = energy.shape
    walks = np.arange(width)          # walk j started at column j
    cols = walks.copy()               # current column of every walk
    paths = np.empty((height, width), dtype=np.int64)
    paths[0] = cols
    totals = energy[0].astype(np.float64)  # total energy of every walk so far

    for r in range(1, height):
        # the 3 candidates of every walk, in tie-break order: left, down, right
        left = np.maximum(cols - 1, 0)
        right = np.minimum(cols + 1, width - 1)
        candidates = np.stack([left, cols, right])            # shape (3, W)
        pick = np.argmin(energy[r, candidates], axis=0)      # first minimum wins
        cols = candidates[pick, walks]
        paths[r] = cols
        totals += energy[r, cols]

    best = int(np.argmin(totals))  # cheapest walk (ties: leftmost start)
    return paths[:, best].copy()


def _make_frame(image: np.ndarray, seam: np.ndarray, step: int, total: int) -> np.ndarray:
    """Copy of the current image with the seam in red and a progress label.
    Only used for the GUI / --visualize; it is not part of the algorithm."""
    frame = draw_seam(image, seam, color=(0, 0, 255))
    label = f"Seam {step}/{total}  size: {image.shape[1]}x{image.shape[0]}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class GreedySeamCarver(SeamCarver):
    """Vertical seam removal using a greedy local choice (not guaranteed optimal)."""

    name = "greedy"
    description = "Greedy seam carving"

    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        """Remove ``num_seams`` vertical seams.

        Each iteration: recompute energy -> find greedy seam -> remove it.
        Energy must be recomputed because removing a seam changes the
        neighbours (and so the gradients) of the pixels next to it.

        Optional keyword arguments (the same ones DP accepts, used by main.py
        and gui.py):
          step_callback   function called after each seam is found, with a dict
                          {step, total, seam, seam_original, frame, image}
          visualize       show each step in an OpenCV window (main.py --visualize)
          step_delay      milliseconds to wait between visualized steps
          show_all_seams  also return all removed seams drawn on the original
                          image in metadata["all_seams_image"]
          multi_start     use find_multistart_greedy_seam instead of the
                          normal single-start greedy (default False)
        """
        if not 0 <= num_seams < image.shape[1]:
            raise ValueError(f"num_seams must be in [0, {image.shape[1] - 1}]")

        step_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("step_callback")
        visualize: bool = kwargs.get("visualize", False)
        step_delay: int = int(kwargs.get("step_delay", 1))
        show_all_seams: bool = kwargs.get("show_all_seams", visualize)
        window_name: str = kwargs.get("window_name", "Seam Carving - Step by Step")
        multi_start: bool = kwargs.get("multi_start", False)
        find_seam = find_multistart_greedy_seam if multi_start else find_greedy_seam

        img = image.copy()
        height, width = image.shape[:2]
        # col_map[r, c] = column in the ORIGINAL image of the pixel now at (r, c).
        # We remove the same seam from it, so we can draw every seam on the original.
        col_map = np.tile(np.arange(width), (height, 1))

        seam_energies: list[float] = []
        seams: list[np.ndarray] = []           # seam in the image it was removed from
        seams_original: list[np.ndarray] = []  # same seam in original-image columns
        elapsed = TimerResult()
        # Time spent in each part of the loop, summed over all seams (seconds)
        energy_s = seam_s = removal_s = 0.0

        with timer(elapsed):
            for step in range(1, num_seams + 1):
                t0 = time.perf_counter()
                energy = sobel_edge_map(img)  # same energy function as DP
                t1 = time.perf_counter()
                seam = find_seam(energy)
                t2 = time.perf_counter()
                energy_s += t1 - t0
                seam_s += t2 - t1

                seam_energies.append(seam_energy(energy, seam))
                seams.append(seam)
                seams_original.append(col_map[np.arange(height), seam])

                # Drawing is only done when someone wants to see it, so normal
                # runs (and the Q3 timings) don't pay for it.
                if step_callback or visualize:
                    frame = _make_frame(img, seam, step, num_seams)
                    if step_callback:
                        step_callback({
                            "step": step,
                            "total": num_seams,
                            "seam": seam.copy(),
                            "seam_original": seams_original[-1],
                            "frame": frame,
                            "image": img.copy(),
                        })
                    if visualize:
                        cv2.imshow(window_name, fit_to_window(frame))
                        cv2.waitKey(step_delay)

                t3 = time.perf_counter()
                img = remove_vertical_seam(img, seam)
                removal_s += time.perf_counter() - t3
                col_map = remove_vertical_seam(col_map, seam)

        all_seams_image = draw_seams(image, seams_original) if show_all_seams else None
        if visualize and all_seams_image is not None:
            cv2.imshow("All Seams (on original)", fit_to_window(all_seams_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return AlgorithmResult(
            image=img,
            algorithm="greedy_multi" if multi_start else self.name,
            runtime_s=elapsed.elapsed_s,
            metadata={
                "seams_removed": num_seams,
                "seam_energies": seam_energies,
                "seams": seams,
                "seams_original": seams_original,
                "all_seams_image": all_seams_image,
                # Breakdown of runtime_s. "other_s" is everything else in the
                # loop: bookkeeping, tracking original columns, GUI drawing.
                "timing": {
                    "energy_s": energy_s,
                    "seam_s": seam_s,
                    "removal_s": removal_s,
                    "other_s": elapsed.elapsed_s - energy_s - seam_s - removal_s,
                },
            },
        )
