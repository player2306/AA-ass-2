"""Tests for the greedy seam finder.

Run from the project root:  python -m tests.test_greedy
"""

import numpy as np

from src.algorithms.greedy import (
    GreedySeamCarver,
    find_greedy_seam,
    find_multistart_greedy_seam,
)
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
    # timing breakdown: every part is non-negative and they add up to the total
    timing = result.metadata["timing"]
    assert all(timing[k] >= 0 for k in ("energy_s", "seam_s", "removal_s"))
    assert abs(sum(timing.values()) - result.runtime_s) < 1e-9


def test_multistart_greedy():
    rng = np.random.default_rng(4)
    for _ in range(200):
        energy = rng.integers(0, 10, size=(6, 5)).astype(float)
        multi = seam_energy(energy, find_multistart_greedy_seam(energy))
        single = seam_energy(energy, find_greedy_seam(energy))
        # same as trying every start one by one with the normal greedy
        every_start = min(seam_energy(energy, find_greedy_seam(energy, start=j))
                          for j in range(energy.shape[1]))
        assert multi == every_start
        assert best_seam_energy(energy) <= multi <= single

    # on the counterexample, starting elsewhere avoids the trap
    energy = np.array([[1, 2, 2],
                       [1, 5, 1],
                       [100, 100, 1]], dtype=float)
    assert seam_energy(energy, find_multistart_greedy_seam(energy)) == 4

    # edge case: width 1, and run() with multi_start
    assert find_multistart_greedy_seam(np.array([[5.0], [3.0]])).tolist() == [0, 0]
    image = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    result = GreedySeamCarver().run(image, num_seams=3, multi_start=True)
    assert result.image.shape == (20, 27, 3) and result.algorithm == "greedy_multi"


def test_dp_matches_brute_force():
    """DP (Member 1's code) must always find the optimal seam. Skipped if DP is missing."""
    try:
        from src.algorithms.dp import find_dp_seam
    except ImportError:
        try:
            from src.algorithms.dp import _find_vertical_seam as find_dp_seam
        except ImportError:
            print("SKIP test_dp_matches_brute_force: no DP seam finder in dp.py")
            return

    rng = np.random.default_rng(5)
    for _ in range(300):
        h, w = rng.integers(1, 7), rng.integers(1, 6)
        energy = rng.integers(0, 10, size=(h, w)).astype(np.float32)
        seam = find_dp_seam(energy)
        assert len(seam) == h and np.all(np.abs(np.diff(seam)) <= 1)   # valid seam
        assert seam_energy(energy, seam) == best_seam_energy(energy)    # optimal
        # and greedy is never better than DP
        assert seam_energy(energy, find_greedy_seam(energy)) >= seam_energy(energy, seam)


def test_seam_similarity():
    from compare_q3 import seam_similarity
    sim = seam_similarity(np.array([1, 2, 3, 3]), np.array([1, 2, 5, 4]))
    assert sim["same_rows_pct"] == 50.0          # rows 0 and 1 match
    assert sim["mean_col_distance"] == 0.75      # gaps 0, 0, 2, 1
    assert sim["max_col_distance"] == 2


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")
