"""Algorithm registry. Members register their implementation here."""

from .base import AlgorithmResult, SeamCarver
from .dp import DPSeamCarver
from .graphcut import GraphCutSeamCarver
from .greedy import GreedySeamCarver

ALGORITHMS: dict[str, type[SeamCarver]] = {
    DPSeamCarver.name: DPSeamCarver,
    GreedySeamCarver.name: GreedySeamCarver,
    GraphCutSeamCarver.name: GraphCutSeamCarver,
}


def get_algorithm(name: str) -> SeamCarver:
    """Create an algorithm instance by its registered name."""
    try:
        return ALGORITHMS[name]()
    except KeyError:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Choose one of: {available}")
