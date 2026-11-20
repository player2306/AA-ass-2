"""Graph cut / max-flow seam carving (Member 4: graph cut).

Seam finding is reformulated as a minimum s-t cut and solved with max-flow.

Graph for a vertical seam on an H x W cost map ``cost`` (one node per pixel):

* ``source -> (i, 0)``            capacity INF  (first column is always on the source side)
* ``(i, j) -> (i, j + 1)``        capacity ``cost[i, j]`` (cutting it removes pixel (i, j))
* ``(i, j + 1) -> (i, j)``        capacity INF  (source side of every row is a prefix,
                                                  so each row is cut exactly once)
* ``(i, j) -> (i - 1, j - 1)``    capacity INF  (neighbouring rows are cut at most
* ``(i, j) -> (i + 1, j - 1)``    capacity INF   one column apart -> connected seam)
* ``(i, W - 1) -> sink``          capacity ``cost[i, W - 1]``

The assignment links the last column to the sink with INF, which means the last
column can never be removed. Using ``cost[i, W - 1]`` instead is the same as padding
the image with one extra column that is joined to the sink with INF, so every column
(including the last) can be part of the seam, exactly as in the DP version.

A finite cut keeps, in each row i, the prefix ``(i, 0) .. (i, p_i)`` on the source
side, with ``|p_i - p_{i+1}| <= 1``. Its capacity is ``sum_i cost[i, p_i]``, the cost of
the seam ``p``. So the minimum cut is the minimum-cost seam, and by the max-flow
min-cut theorem its capacity equals the max-flow value. After the max-flow, a BFS
from the source in the residual graph gives the source side S, and the rightmost
pixel of S in each row is the seam pixel to remove.

Horizontal seams are found the same way on the transposed image.
"""

from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from ..display import fit_to_window
from ..utils import draw_seam, draw_seams, remove_vertical_seam, sobel_edge_map
from ..utils.timing import TimerResult, timer
from .base import AlgorithmResult, SeamCarver
from .both_directions import carve_both_directions

INF = float("inf")

MAX_FLOW_METHODS = ("dinic", "edmonds_karp")
ENERGY_MODES = ("backward", "forward")
DIRECTIONS = ("vertical", "horizontal")


# --------------------------------------------------------------------------- #
# Generic max-flow solver
# --------------------------------------------------------------------------- #
class FlowNetwork:
    """Directed flow network stored as a residual graph in CSR form.

    Every edge ``u -> v`` is stored as two arcs: the forward arc with residual
    capacity ``cap`` and the reverse arc ``v -> u`` with residual capacity
    ``rev_cap`` (0 for an ordinary edge). Pushing ``f`` units along an arc lowers
    its residual capacity by ``f`` and raises its reverse arc's by ``f``.

    A pair of opposite edges (like ``(i, j) -> (i, j + 1)`` with capacity ``e`` and
    ``(i, j + 1) -> (i, j)`` with INF) is stored as ONE arc pair with
    ``cap = e`` and ``rev_cap = INF``. That models the same residual network with
    half the arcs.

    Arc arrays (Python lists, because indexing a list is much faster than indexing
    a numpy array one element at a time):

    * ``head[a]``  node the arc points to
    * ``cap[a]``   residual capacity of the arc
    * ``rev[a]``   index of the reverse arc
    * ``first[u]`` arcs leaving node ``u`` are ``first[u] .. first[u + 1] - 1``
    """

    def __init__(
        self,
        num_nodes: int,
        tails: np.ndarray,
        heads: np.ndarray,
        caps: np.ndarray,
        rev_caps: np.ndarray | None = None,
    ):
        tails = np.asarray(tails, dtype=np.int64).ravel()
        heads = np.asarray(heads, dtype=np.int64).ravel()
        caps = np.asarray(caps, dtype=np.float64).ravel()
        rev_caps = np.zeros_like(caps) if rev_caps is None else np.asarray(rev_caps, dtype=np.float64).ravel()
        if not (len(tails) == len(heads) == len(caps) == len(rev_caps)):
            raise ValueError("tails, heads, caps and rev_caps must have the same length")
        if (caps < 0).any() or (rev_caps < 0).any():
            raise ValueError("capacities must be non-negative")

        m = len(tails)
        # Arc k is edge k, arc k + m is its reverse.
        arc_tail = np.concatenate([tails, heads])
        arc_head = np.concatenate([heads, tails])
        arc_cap = np.concatenate([caps, rev_caps])
        arc_pair = np.concatenate([np.arange(m, 2 * m), np.arange(m)])

        # Sort the arcs by tail node so the arcs of each node are contiguous (CSR).
        order = np.argsort(arc_tail, kind="stable")
        new_index = np.empty(2 * m, dtype=np.int64)
        new_index[order] = np.arange(2 * m)

        self.num_nodes = num_nodes
        self.num_edges = m
        self.head: list[int] = arc_head[order].tolist()
        self.cap: list[float] = arc_cap[order].tolist()
        self.rev: list[int] = new_index[arc_pair[order]].tolist()
        self.first: list[int] = np.searchsorted(arc_tail[order], np.arange(num_nodes + 1)).tolist()

    @classmethod
    def from_edges(cls, num_nodes: int, edges: list[tuple[int, int, float]]) -> "FlowNetwork":
        """Build a network from a plain list of ``(u, v, capacity)`` edges."""
        if not edges:
            return cls(num_nodes, [], [], [])
        tails, heads, caps = zip(*edges)
        return cls(num_nodes, tails, heads, caps)

    # ----------------------------------------------------------------- max-flow
    def max_flow(self, source: int, sink: int, method: str = "dinic") -> tuple[float, dict[str, int]]:
        """Push the maximum flow from ``source`` to ``sink``.

        Returns ``(flow_value, stats)``. The residual capacities are updated in
        place, so ``min_cut_source_side`` can be called afterwards.
        """
        if source == sink:
            raise ValueError("source and sink must differ")
        if method == "dinic":
            return self._dinic(source, sink)
        if method == "edmonds_karp":
            return self._edmonds_karp(source, sink)
        raise ValueError(f"unknown max-flow method '{method}', choose one of {MAX_FLOW_METHODS}")

    def _edmonds_karp(self, s: int, t: int) -> tuple[float, dict[str, int]]:
        """Ford-Fulkerson that always augments along a shortest path (found by BFS).

        O(V E) augmentations, each needing an O(E) BFS -> O(V E^2) time.
        """
        head, cap, rev, first = self.head, self.cap, self.rev, self.first
        n = self.num_nodes
        flow = 0.0
        paths = 0

        while True:
            # BFS from s; parent_arc[v] is the arc used to reach v.
            parent_arc = [-1] * n
            parent_arc[s] = -2
            queue = deque([s])
            while queue and parent_arc[t] == -1:
                u = queue.popleft()
                for a in range(first[u], first[u + 1]):
                    if cap[a] > 0:
                        v = head[a]
                        if parent_arc[v] == -1:
                            parent_arc[v] = a
                            queue.append(v)
            if parent_arc[t] == -1:  # no augmenting path left -> flow is maximum
                break

            # Walk back from t to s to collect the path and its bottleneck.
            path = []
            v = t
            while v != s:
                a = parent_arc[v]
                path.append(a)
                v = head[rev[a]]  # tail of a = head of its reverse arc
            bottleneck = min(cap[a] for a in path)

            for a in path:
                cap[a] -= bottleneck
                cap[rev[a]] += bottleneck
            flow += bottleneck
            paths += 1

        return flow, {"bfs_runs": paths + 1, "augmenting_paths": paths}

    def _bfs_levels(self, s: int, t: int) -> list[int]:
        """Level graph for Dinic: ``level[v]`` = BFS distance from ``s`` in the residual graph."""
        head, cap, first = self.head, self.cap, self.first
        level = [-1] * self.num_nodes
        level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if level[t] >= 0 and level[u] >= level[t]:
                break  # nodes this far away cannot be on a shortest s-t path
            next_level = level[u] + 1
            for a in range(first[u], first[u + 1]):
                if cap[a] > 0:
                    v = head[a]
                    if level[v] < 0:
                        level[v] = next_level
                        queue.append(v)
        return level

    def _dinic(self, s: int, t: int) -> tuple[float, dict[str, int]]:
        """Dinic's algorithm: BFS level graph + blocking flow found by DFS.

        Each phase finds every shortest augmenting path at once, and the s-t
        distance strictly grows between phases -> at most V phases, O(V^2 E) time.
        """
        head, cap, rev, first = self.head, self.cap, self.rev, self.first
        flow = 0.0
        phases = 0
        paths = 0

        while True:
            level = self._bfs_levels(s, t)
            if level[t] < 0:  # sink unreachable -> flow is maximum
                break
            phases += 1
            current = first[:]  # current-arc pointer of each node

            # Iterative DFS. nodes[k] is the k-th node on the path, arcs[k] leaves it.
            nodes = [s]
            arcs: list[int] = []
            while nodes:
                u = nodes[-1]
                if u == t:
                    bottleneck = min(cap[a] for a in arcs)
                    for a in arcs:
                        cap[a] -= bottleneck
                        cap[rev[a]] += bottleneck
                    flow += bottleneck
                    paths += 1
                    # Back up to the tail of the first saturated arc and continue from there.
                    k = 0
                    while cap[arcs[k]] > 0:
                        k += 1
                    del arcs[k:]
                    del nodes[k + 1 :]
                    continue

                # Advance along the first admissible arc (positive residual, next level).
                a = current[u]
                end = first[u + 1]
                want = level[u] + 1
                while a < end and not (cap[a] > 0 and level[head[a]] == want):
                    a += 1
                current[u] = a
                if a < end:
                    arcs.append(a)
                    nodes.append(head[a])
                else:
                    # Dead end: drop u from the level graph and retreat.
                    level[u] = -1
                    nodes.pop()
                    if arcs:
                        arcs.pop()

        return flow, {"bfs_runs": phases + 1, "phases": phases, "augmenting_paths": paths}

    # ------------------------------------------------------------------ min-cut
    def min_cut_source_side(self, source: int) -> np.ndarray:
        """Nodes reachable from ``source`` in the residual graph (the set S of the min cut).

        Call this after ``max_flow``. Returns a boolean array of length ``num_nodes``.
        """
        head, cap, first = self.head, self.cap, self.first
        seen = [False] * self.num_nodes
        seen[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for a in range(first[u], first[u + 1]):
                if cap[a] > 0:
                    v = head[a]
                    if not seen[v]:
                        seen[v] = True
                        queue.append(v)
        return np.array(seen, dtype=bool)


# --------------------------------------------------------------------------- #
# Seam graph construction and seam extraction
# --------------------------------------------------------------------------- #
def compute_cost_map(image: np.ndarray, energy: str = "backward") -> np.ndarray:
    """Cost of removing each pixel, used as the capacity of its right-going edge.

    * ``backward``: Sobel energy ``|Gx| + |Gy|`` (the same energy map as the DP version).
    * ``forward``:  ``|I[i, j+1] - I[i, j-1]|``, the new edge created between the left
      and right neighbours when pixel (i, j) is removed (border pixels are replicated).

    Both are integer-valued for 8-bit images, so the max-flow uses integer capacities.
    """
    if energy == "backward":
        return sobel_edge_map(image).astype(np.float64)
    if energy == "forward":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        padded = np.pad(gray.astype(np.float64), ((0, 0), (1, 1)), mode="edge")
        return np.abs(padded[:, 2:] - padded[:, :-2])
    raise ValueError(f"unknown energy '{energy}', choose one of {ENERGY_MODES}")


def build_seam_graph(cost: np.ndarray) -> tuple[FlowNetwork, int, int]:
    """Build the s-t graph whose minimum cut is the minimum-cost vertical seam.

    Pixel (i, j) is node ``i * W + j``; the source is node ``H * W`` and the sink
    is node ``H * W + 1``. Returns ``(network, source, sink)``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.size == 0:
        raise ValueError("cost must be a non-empty 2-D array")
    if (cost < 0).any() or not np.isfinite(cost).all():
        raise ValueError("cost must be finite and non-negative")

    h, w = cost.shape
    source, sink = h * w, h * w + 1
    node = np.arange(h * w).reshape(h, w)

    tails: list[np.ndarray] = []
    heads: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    rev_caps: list[np.ndarray] = []

    def add_edges(u, v, cap, rev_cap=0.0):
        u, v, cap, rev_cap = np.broadcast_arrays(u, v, cap, rev_cap)
        tails.append(u.ravel())
        heads.append(v.ravel())
        caps.append(cap.ravel())
        rev_caps.append(rev_cap.ravel())

    # source -> first column (INF)
    add_edges(source, node[:, 0], INF)
    # (i, j) -> (i, j+1) with the cost of removing (i, j); the reverse arc is the
    # INF backward edge (i, j+1) -> (i, j), stored in the same arc pair.
    add_edges(node[:, :-1], node[:, 1:], cost[:, :-1], INF)
    # last column -> sink with the cost of removing (i, W-1)
    add_edges(node[:, -1], sink, cost[:, -1])
    # diagonal INF edges (i, j) -> (i-1, j-1) and (i, j) -> (i+1, j-1)
    add_edges(node[1:, 1:], node[:-1, :-1], INF)
    add_edges(node[:-1, 1:], node[1:, :-1], INF)

    network = FlowNetwork(
        h * w + 2,
        np.concatenate(tails),
        np.concatenate(heads),
        np.concatenate(caps),
        np.concatenate(rev_caps),
    )
    return network, source, sink


def find_graphcut_seam(
    cost: np.ndarray,
    method: str = "dinic",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Find the minimum-cost vertical seam with min-cut / max-flow.

    1. Build the seam graph.
    2. Run max-flow from the source to the sink.
    3. BFS from the source in the residual graph -> source side S of the min cut.
    4. In each row, the rightmost pixel in S is the seam pixel.

    Returns ``seam`` (``seam[i]`` = column removed from row ``i``), or
    ``(seam, info)`` when ``return_info`` is True.
    """
    h, w = np.shape(cost)
    network, source, sink = build_seam_graph(cost)
    flow, stats = network.max_flow(source, sink, method=method)

    in_source_side = network.min_cut_source_side(source)[: h * w].reshape(h, w)
    # Every row starts on the source side (INF edge from the source), so this is
    # well defined: index of the last True value in each row.
    seam = (w - 1 - np.argmax(in_source_side[:, ::-1], axis=1)).astype(np.int32)

    if not return_info:
        return seam
    info = {
        "max_flow": flow,
        "nodes": network.num_nodes,
        "edges": network.num_edges,
        "method": method,
        **stats,
    }
    return seam, info


def find_graphcut_horizontal_seam(
    cost: np.ndarray,
    method: str = "dinic",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Horizontal seam = vertical seam of the transposed cost map.

    ``seam[j]`` is the row removed from column ``j``.
    """
    return find_graphcut_seam(np.asarray(cost).T, method=method, return_info=return_info)


def seam_cost(cost: np.ndarray, seam: np.ndarray) -> float:
    """Total cost of a vertical seam: ``sum_i cost[i, seam[i]]``."""
    return float(np.asarray(cost)[np.arange(len(seam)), seam].sum())


# --------------------------------------------------------------------------- #
# Seam carver
# --------------------------------------------------------------------------- #
def _transpose_if_horizontal(image: np.ndarray, direction: str) -> np.ndarray:
    """Transpose the image for horizontal seams so the rest of the code only handles vertical ones.

    Transposing twice gives the original back, so the same call also undoes it.
    """
    if direction == "vertical":
        return image
    axes = (1, 0, 2) if image.ndim == 3 else (1, 0)
    # .copy(), not np.ascontiguousarray: for a 1-row image ascontiguousarray returns a
    # view with strides OpenCV rejects (e.g. putText on step frames fails)
    return np.transpose(image, axes).copy()


def _show_step(frame: np.ndarray, title: str, delay: int = 1) -> int:
    cv2.imshow(title, fit_to_window(frame))
    return cv2.waitKey(delay) & 0xFF


class GraphCutSeamCarver(SeamCarver):
    """Optimal seam removal formulated as a min-cut / max-flow problem."""

    name = "graphcut"
    description = "Graph cut (min-cut / max-flow) seam carving"

    def run(
        self,
        image: np.ndarray,
        num_seams: int = 1,
        **kwargs: Any,
    ) -> AlgorithmResult:
        """Remove ``num_seams`` seams. Each iteration: cost map -> max-flow -> min cut -> remove seam.

        Extra keyword arguments:

        * ``direction``: ``"vertical"`` (default, narrower image), ``"horizontal"`` (shorter image)
          or ``"both"`` (``num_seams`` vertical then ``num_seams`` horizontal, see both_directions.py)
        * ``energy``: ``"backward"`` (default, Sobel, same as DP) or ``"forward"``
        * ``method``: ``"dinic"`` (default) or ``"edmonds_karp"``
        * ``visualize``, ``step_delay``, ``show_all_seams``, ``window_name``,
          ``step_callback``, ``keep_steps``: same meaning as in the DP version
        """
        direction: str = kwargs.get("direction", "vertical")
        energy_mode: str = kwargs.get("energy", "backward")
        method: str = kwargs.get("method", "dinic")
        if direction == "both":
            # vertical seams first, then horizontal; energy/method are passed on to both batches
            return carve_both_directions(self, image, num_seams, **kwargs)
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS + ('both',)}")
        if energy_mode not in ENERGY_MODES:
            raise ValueError(f"energy must be one of {ENERGY_MODES}")
        if method not in MAX_FLOW_METHODS:
            raise ValueError(f"method must be one of {MAX_FLOW_METHODS}")

        work = _transpose_if_horizontal(image, direction).copy()
        if num_seams < 0:
            raise ValueError("num_seams must be >= 0")
        if num_seams >= work.shape[1]:
            side = "width" if direction == "vertical" else "height"
            raise ValueError(f"cannot remove more seams than image {side}")

        visualize: bool = kwargs.get("visualize", False)
        step_delay: int = int(kwargs.get("step_delay", 1))
        show_all_seams: bool = kwargs.get("show_all_seams", visualize)
        window_name: str = kwargs.get("window_name", "Graph Cut Seam Carving - Step by Step")
        step_callback: Callable[[dict[str, Any]], None] | None = kwargs.get("step_callback")
        keep_steps: bool = kwargs.get("keep_steps", False)
        need_frames = visualize or step_callback is not None or keep_steps

        original = work.copy()
        h, w = work.shape[:2]
        col_map = np.broadcast_to(np.arange(w), (h, w)).copy()  # original column of every pixel
        seams: list[np.ndarray] = []
        seams_original: list[np.ndarray] = []
        seam_energies: list[float] = []
        solver_stats: list[dict[str, Any]] = []
        step_frames: list[np.ndarray] = []
        elapsed = TimerResult()

        with timer(elapsed):
            for i in range(num_seams):
                cost = compute_cost_map(work, energy_mode)
                seam, info = find_graphcut_seam(cost, method=method, return_info=True)
                seam_energies.append(seam_cost(cost, seam))
                solver_stats.append(info)
                seams.append(seam.copy())
                seam_orig = col_map[np.arange(h), seam].copy()
                seams_original.append(seam_orig)

                if need_frames:
                    frame = _transpose_if_horizontal(draw_seam(work, seam, color=(0, 0, 255)), direction)
                    cv2.putText(
                        frame,
                        f"Seam {i + 1}/{num_seams}  size: {frame.shape[1]}x{frame.shape[0]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if keep_steps:
                        step_frames.append(frame.copy())
                    if step_callback:
                        step_callback(
                            {
                                "step": i + 1,
                                "total": num_seams,
                                "seam": seam.copy(),
                                "seam_original": seam_orig,
                                "frame": frame,
                                "image": _transpose_if_horizontal(work, direction).copy(),
                            }
                        )
                    if visualize:
                        _show_step(frame, window_name, delay=step_delay)

                work = remove_vertical_seam(work, seam)
                keep = np.ones(col_map.shape, dtype=bool)
                keep[np.arange(h), seam] = False
                col_map = col_map[keep].reshape(h, -1)

        all_seams_image = None
        if show_all_seams:
            overlay = draw_seams(original, seams_original, color=(0, 0, 255))
            all_seams_image = _transpose_if_horizontal(overlay, direction)
        if visualize and all_seams_image is not None:
            cv2.imshow("All Seams (on original)", fit_to_window(all_seams_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return AlgorithmResult(
            image=_transpose_if_horizontal(work, direction),
            algorithm=self.name,
            runtime_s=elapsed.elapsed_s,
            metadata={
                "seams_removed": num_seams,
                "direction": direction,
                "energy": energy_mode,
                "method": method,
                "seams": seams,
                "seams_original": seams_original,
                "seam_energies": seam_energies,
                "max_flows": [s["max_flow"] for s in solver_stats],
                "solver_stats": solver_stats,
                "all_seams_image": all_seams_image,
                "step_frames": step_frames,
            },
        )
