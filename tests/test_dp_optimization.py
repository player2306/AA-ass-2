"""Regression checks for the rolling-row DP implementation."""

import numpy as np

from src.algorithms.dp import _find_vertical_seam


def _full_table_seam(energy):
    """Previous full-table algorithm, retained as a regression reference."""
    h, w = energy.shape
    costs = energy.copy()
    parents = np.zeros((h, w), dtype=np.int32)
    cols = np.arange(w)
    for row in range(1, h):
        left = np.roll(costs[row - 1], 1)
        left[0] = np.inf
        right = np.roll(costs[row - 1], -1)
        right[-1] = np.inf
        choices = np.vstack((left, costs[row - 1], right))
        offsets = np.argmin(choices, axis=0) - 1
        parents[row] = cols + offsets
        costs[row] += choices[offsets + 1, cols]

    seam = np.empty(h, dtype=np.int32)
    seam[-1] = np.argmin(costs[-1])
    for row in range(h - 2, -1, -1):
        seam[row] = parents[row + 1, seam[row + 1]]
    return seam


def test_rolling_costs_match_full_table():
    rng = np.random.default_rng(3151)
    for h, w in [(1, 1), (1, 20), (20, 1), (2, 2), (9, 13), (53, 67)]:
        for _ in range(20):
            # Integer-valued energies exercise ties and boundary choices.
            energy = rng.integers(0, 10, size=(h, w)).astype(np.float32)
            actual = _find_vertical_seam(energy)
            assert np.array_equal(actual, _full_table_seam(energy))
            assert np.all(np.abs(np.diff(actual)) <= 1)
            assert np.all((0 <= actual) & (actual < w))


if __name__ == "__main__":
    test_rolling_costs_match_full_table()
    print("PASS test_rolling_costs_match_full_table")
