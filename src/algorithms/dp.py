"""Dynamic programming seam carving (Member 1: DP core engine)."""

from typing import Any

import numpy as np

from .base import AlgorithmResult, SeamCarver


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
        raise NotImplementedError(
            "DP seam carving is not implemented yet. "
            "Member 1: add energy computation and the DP seam search here."
        )
