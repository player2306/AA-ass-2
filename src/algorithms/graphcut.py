"""Graph cut / max-flow seam carving (Member 4: graph cut)."""

from typing import Any

import numpy as np

from .base import AlgorithmResult, SeamCarver


class GraphCutSeamCarver(SeamCarver):
    """Vertical seam removal formulated as a min-cut / max-flow problem."""

    name = "graphcut"
    description = "Graph cut (max-flow) seam carving"

    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        raise NotImplementedError(
            "Graph cut seam carving is not implemented yet. "
            "Member 4: add the graph construction and max-flow solver here."
        )
