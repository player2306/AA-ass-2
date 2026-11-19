"""Tests for the graph cut (min-cut / max-flow) seam finder.

Run from the project root:  python -m tests.test_graphcut
(also works with pytest:     python -m pytest tests/test_graphcut.py)
"""

import numpy as np

from src.algorithms.dp import _find_vertical_seam as find_dp_seam
from src.algorithms.graphcut import (
    MAX_FLOW_METHODS,
    FlowNetwork,
    GraphCutSeamCarver,
    build_seam_graph,
    compute_cost_map,
    find_graphcut_horizontal_seam,
    find_graphcut_seam,
    seam_cost,
)


def all_seams(height: int, width: int):
    """Every connected vertical seam (exponentially many, tiny sizes only)."""
    for start in range(width):
        partial = [[start]]
        for _ in range(height - 1):
            partial = [s + [c] for s in partial for c in (s[-1] - 1, s[-1], s[-1] + 1) if 0 <= c < width]
        yield from partial


def best_seam_cost(cost: np.ndarray) -> float:
    """Brute force: the true optimum that DP and graph cut must both find."""
    h, w = cost.shape
    return min(seam_cost(cost, np.array(s)) for s in all_seams(h, w))


def assert_valid_seam(seam: np.ndarray, height: int, width: int) -> None:
    assert len(seam) == height
    assert seam.min() >= 0 and seam.max() < width          # stays in the image
    assert np.all(np.abs(np.diff(seam)) <= 1)              # connected (vertical/diagonal steps)


def test_textbook_max_flow():
    # CLRS Figure 26.1 network: max flow is 23.
    edges = [(0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (2, 4, 14),
             (3, 2, 9), (3, 5, 20), (4, 3, 7), (4, 5, 4)]
    for method in MAX_FLOW_METHODS:
        net = FlowNetwork.from_edges(6, edges)
        flow, _ = net.max_flow(0, 5, method=method)
        assert flow == 23
        side = net.min_cut_source_side(0)
        cut = sum(c for u, v, c in edges if side[u] and not side[v])
        assert cut == 23                                     # max-flow = min-cut


def test_hand_traced_example():
    cost = np.array([[1, 2, 2],
                     [1, 5, 1],
                     [100, 100, 1]], dtype=float)
    for method in MAX_FLOW_METHODS:
        seam, info = find_graphcut_seam(cost, method=method, return_info=True)
        assert seam.tolist() == [1, 2, 2]                    # 2 + 1 + 1 = 4
        assert info["max_flow"] == 4 == seam_cost(cost, seam)


def test_last_and_first_column_can_be_removed():
    cost = np.full((4, 5), 9.0)
    cost[:, -1] = 0
    assert find_graphcut_seam(cost).tolist() == [4, 4, 4, 4]
    cost = np.full((4, 5), 9.0)
    cost[:, 0] = 0
    assert find_graphcut_seam(cost).tolist() == [0, 0, 0, 0]


def test_degenerate_sizes():
    assert find_graphcut_seam(np.array([[5.0], [3.0], [7.0]])).tolist() == [0, 0, 0]   # width 1
    assert find_graphcut_seam(np.array([[4.0, 1.0, 3.0]])).tolist() == [1]              # height 1
    assert find_graphcut_seam(np.zeros((3, 3))).tolist() == [0, 0, 0]                  # all zero


def test_matches_brute_force_on_random_input():
    rng = np.random.default_rng(0)
    for _ in range(150):
        h, w = rng.integers(1, 6), rng.integers(1, 6)
        cost = rng.integers(0, 10, size=(h, w)).astype(float)
        best = best_seam_cost(cost)
        for method in MAX_FLOW_METHODS:
            seam, info = find_graphcut_seam(cost, method=method, return_info=True)
            assert_valid_seam(seam, h, w)
            assert info["max_flow"] == best                  # max-flow value = optimal seam cost
            assert seam_cost(cost, seam) == best             # and the extracted seam achieves it


def test_every_seam_is_a_finite_cut():
    # Each seam corresponds to a cut with capacity == seam cost (bijection argument).
    rng = np.random.default_rng(3)
    cost = rng.integers(0, 10, size=(4, 4)).astype(float)
    net, s, t = build_seam_graph(cost)
    h, w = cost.shape
    for seam in all_seams(h, w):
        in_s = np.zeros(h * w + 2, dtype=bool)
        in_s[s] = True
        for i, p in enumerate(seam):
            in_s[i * w : i * w + p + 1] = True
        cut = 0.0
        for u in range(net.num_nodes):
            for a in range(net.first[u], net.first[u + 1]):
                if in_s[u] and not in_s[net.head[a]]:
                    cut += net.cap[a]
        assert cut == seam_cost(cost, np.array(seam))


def test_matches_dp_on_larger_input():
    rng = np.random.default_rng(1)
    for _ in range(5):
        cost = rng.integers(0, 50, size=(30, 40)).astype(np.float32)
        dp = find_dp_seam(cost)
        gc, info = find_graphcut_seam(cost, return_info=True)
        assert_valid_seam(gc, 30, 40)
        assert seam_cost(cost, gc) == seam_cost(cost, dp) == info["max_flow"]


def test_horizontal_seam():
    rng = np.random.default_rng(2)
    cost = rng.integers(0, 10, size=(6, 8)).astype(float)
    seam = find_graphcut_horizontal_seam(cost)
    assert_valid_seam(seam, 8, 6)                            # one row index per column
    assert cost[seam, np.arange(8)].sum() == best_seam_cost(cost.T)


def test_cost_maps():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[:, 3:] = 255                                       # vertical edge between col 2 and 3
    forward = compute_cost_map(image, "forward")
    # removing col 2 or col 3 joins a black pixel with a white one
    assert forward[0].tolist() == [0, 0, 255, 255, 0]
    backward = compute_cost_map(image, "backward")
    assert backward.shape == (3, 5) and backward[:, 0].sum() == 0


def test_run_vertical_horizontal_and_forward():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    carver = GraphCutSeamCarver()

    result = carver.run(image, num_seams=3, show_all_seams=True)
    assert result.image.shape == (12, 13, 3)
    assert len(result.metadata["seams"]) == 3
    assert result.metadata["seam_energies"] == result.metadata["max_flows"]
    assert result.metadata["all_seams_image"].shape == image.shape
    for i, seam in enumerate(result.metadata["seams"]):
        assert_valid_seam(seam, 12, 16 - i)                # connected in the image it was cut from
    # In original-image columns a later seam may jump 2+ columns where earlier
    # seams were removed, so only check bounds and that no pixel is removed twice.
    removed = np.stack(result.metadata["seams_original"])  # (seams, rows)
    assert removed.min() >= 0 and removed.max() < 16
    assert all(len(set(removed[:, r])) == 3 for r in range(12))
    assert removed.shape == (3, 12)

    result = carver.run(image, num_seams=4, direction="horizontal", method="edmonds_karp")
    assert result.image.shape == (8, 16, 3)

    result = carver.run(image, num_seams=2, energy="forward")
    assert result.image.shape == (12, 14, 3)


def test_run_matches_dp_carver():
    from src.algorithms.dp import DPSeamCarver

    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    gc = GraphCutSeamCarver().run(image, num_seams=1)
    dp = DPSeamCarver().run(image, num_seams=1)
    cost = compute_cost_map(image)
    assert seam_cost(cost, gc.metadata["seams"][0]) == seam_cost(cost, dp.metadata["seams"][0])


def test_invalid_arguments():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for kwargs in ({"num_seams": 4}, {"num_seams": -1}, {"direction": "diagonal"},
                   {"energy": "magic"}, {"method": "push_relabel"}):
        try:
            GraphCutSeamCarver().run(image, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed.")
