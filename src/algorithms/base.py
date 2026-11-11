"""Common interface that every seam carving algorithm implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AlgorithmResult:
    """Output returned by a seam carving algorithm."""

    image: np.ndarray
    algorithm: str = ""
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SeamCarver(ABC):
    """Base class for seam carving implementations.

    Each team member adds a subclass with a unique ``name``. ``run`` receives
    the original BGR image and returns an ``AlgorithmResult`` containing the
    carved image. Algorithms may add their own keyword arguments.
    """

    name = "base"
    description = ""

    @abstractmethod
    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        """Remove ``num_seams`` seams from ``image`` and return the result."""
        raise NotImplementedError
