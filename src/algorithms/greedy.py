"""Greedy seam carving (Member 3: greedy implementation)."""

from typing import Any

import numpy as np

from .base import AlgorithmResult, SeamCarver


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
        raise NotImplementedError(
            "Greedy seam carving is not implemented yet. "
            "Member 3: add the greedy seam selection here."
        )
