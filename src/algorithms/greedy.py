"""Greedy seam carving (Member 3: greedy implementation)."""

from typing import Any

import numpy as np

from ..seam_utils import compute_energy, remove_vertical_seam, seam_energy
from ..utils.timing import TimerResult, timer
from .base import AlgorithmResult, SeamCarver


def find_greedy_seam(energy: np.ndarray) -> np.ndarray:
    """Find a vertical seam with a greedy, row-by-row local choice.

    1. Start at the lowest-energy pixel in the top row.
    2. For every next row, look only at the (up to) 3 pixels below the
       current one: down-left, down, down-right. Move to the cheapest.
    3. Never look further ahead and never undo a choice -> fast but not optimal.

    Boundaries: the candidate window is clamped to [0, width-1], so a seam on
    the left/right edge only has 2 candidates (no wrap-around).
    Ties: np.argmin returns the first minimum, so the leftmost candidate wins.
    This makes the result deterministic.

    Time O(H + W) per seam, extra space O(H).
    """
    height, width = energy.shape
    seam = np.zeros(height, dtype=int)

    seam[0] = np.argmin(energy[0])  # cheapest pixel in the top row

    for r in range(1, height):
        prev = seam[r - 1]
        left = max(prev - 1, 0)
        right = min(prev + 1, width - 1)
        # argmin gives the position inside the slice, so add `left` back
        seam[r] = left + np.argmin(energy[r, left:right + 1])

    return seam


class GreedySeamCarver(SeamCarver):
    """Suboptimal vertical seam removal using a greedy local choice."""

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
        """
        if not 0 <= num_seams < image.shape[1]:
            raise ValueError(f"num_seams must be in [0, {image.shape[1] - 1}]")

        img = image.copy()
        seam_energies = []
        elapsed = TimerResult()

        with timer(elapsed):
            for _ in range(num_seams):
                energy = compute_energy(img)
                seam = find_greedy_seam(energy)
                seam_energies.append(seam_energy(energy, seam))
                img = remove_vertical_seam(img, seam)

        return AlgorithmResult(
            image=img,
            algorithm=self.name,
            runtime_s=elapsed.elapsed_s,
            metadata={"seam_energies": seam_energies},
        )
