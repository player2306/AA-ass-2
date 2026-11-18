"""Seam helpers used by the Greedy code and the Q3 comparison.

The energy map, seam removal and seam drawing now come from the team's shared
code in ``src/utils`` (written by Member 1), so Greedy and DP always use exactly
the same functions. This file re-exports them so older imports such as
``from src.seam_utils import compute_energy`` keep working, and adds
``seam_energy``, which the shared code does not have.

A vertical seam is a 1-D int array ``seam`` of length H, where ``seam[r]`` is
the column of the pixel removed from row ``r``.
"""

import numpy as np

from .utils import draw_seam, remove_vertical_seam, sobel_edge_map

__all__ = ["compute_energy", "remove_vertical_seam", "draw_seam", "seam_energy"]


def compute_energy(image: np.ndarray) -> np.ndarray:
    """Energy map of the image: the shared Sobel |Gx| + |Gy| (float32, shape (H, W)).

    High energy = edges/detail (keep). Low energy = flat areas (safe to remove).
    """
    return sobel_edge_map(image)


def seam_energy(energy: np.ndarray, seam: np.ndarray) -> float:
    """Total energy of a seam (sum of the energy of its pixels). Lower = better seam.

    The sum is done in float64 so that long totals stay accurate even though
    the energy map itself is float32.
    """
    rows = np.arange(len(seam))
    return float(energy[rows, seam].sum(dtype=np.float64))
