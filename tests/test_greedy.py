"""Tests for the greedy seam finder.

Run from the project root:  python -m tests.test_greedy
"""

import numpy as np

from src.algorithms.greedy import GreedySeamCarver, find_greedy_seam
from src.seam_utils import remove_vertical_seam, seam_energy


def best_seam_energy(energy: np.ndarray) -> float:
    """Brute force: try every valid seam and return the lowest total energy.
    Exponential time, only for tiny matrices. Gives the optimum that DP must match."""
    height, width = energy.shape

    def walk(r, c):
        if r == height - 1:
            return energy[r, c]
        nxt = range(max(c - 1, 0), min(c + 1, width - 1) + 1)
        return energy[r, c] + min(walk(r + 1, j) for j in nxt)

    return min(walk(0, c) for c in range(width))


def test_hand_traced_example():
    energy = np.array([[3, 1, 4],
                       [1, 5, 9],
                       [2, 6, 5]], dtype=float)
    # row0: min=1 at col1 -> row1: cols0..2 (1,5,9) -> col0 -> row2: cols0..1 (2,6) -> col0
    assert find_greedy_seam(energy).tolist() == [1, 0, 0]


def test_greedy_worse_than_optimal():
    # Greedy grabs the cheap 1s on the left, then is trapped above the 100s.
    energy = np.array([[1, 2, 2],
                       [1, 5, 1],
                       [100, 100, 1]], dtype=float)
    seam = find_greedy_seam(energy)
    assert seam.tolist() == [0, 0, 0]
    assert seam_energy(energy, seam) == 102   # greedy: 1 + 1 + 100
    assert best_seam_energy(energy) == 4      # optimal (DP): 2 + 1 + 1


def test_seam_is_valid_on_random_input():
    rng = np.random.default_rng(0)
    energy = rng.random((50, 30))
    seam = find_greedy_seam(energy)
    assert len(seam) == 50
    assert seam.min() >= 0 and seam.max() < 30            # stays in the image
    assert np.all(np.abs(np.diff(seam)) <= 1)             # connected seam


def test_greedy_never_beats_optimal():
    rng = np.random.default_rng(1)
    for _ in range(200):
        energy = rng.integers(0, 10, size=(5, 4)).astype(float)
        assert seam_energy(energy, find_greedy_seam(energy)) >= best_seam_energy(energy)


def test_edges_and_ties():
    # width 1: the only seam is column 0
    assert find_greedy_seam(np.array([[5.0], [3.0], [7.0]])).tolist() == [0, 0, 0]
    # all equal: leftmost wins every time (tie-break rule)
    assert find_greedy_seam(np.ones((4, 4))).tolist() == [0, 0, 0, 0]
    # seam stuck on the right edge only has 2 candidates and must not go out of bounds
    energy = np.array([[9, 9, 0],
                       [9, 9, 0],
                       [9, 9, 0]], dtype=float)
    assert find_greedy_seam(energy).tolist() == [2, 2, 2]


def test_remove_seam_and_run():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    out = remove_vertical_seam(image, np.array([0, 1, 3]))
    assert out.shape == (3, 3, 3)
    assert out[0, 0].tolist() == image[0, 1].tolist()   # col 0 removed in row 0

    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    result = GreedySeamCarver().run(image, num_seams=5)
    assert result.image.shape == (20, 25, 3)
    assert len(result.metadata["seam_energies"]) == 5


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")
