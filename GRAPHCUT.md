# Graph Cut Seam Carving (Question 4), by Member 4 (Wee Jie)

This file covers how seam finding is turned into a minimum s–t cut, every function written
for the Graph Cut part, the complexity, how to run it, and how it compares with DP.

A **vertical seam** is stored as a 1-D int array `seam` of length `H` (image height).
`seam[i]` is the column of the pixel removed from row `i`. A horizontal seam is the same
thing on the transposed image (`seam[j]` = row removed from column `j`).

---

## The idea: a seam is a cut

Build a graph with **one node per pixel**, plus a source `s` and a sink `t`. Pixel `(i, j)`
is node `i * W + j`; `s = H*W`, `t = H*W + 1`. `cost[i, j]` is the cost of removing pixel
`(i, j)` (its Sobel energy `e[i, j]`, or the forward energy `|I[i, j+1] - I[i, j-1]|`).

| Edge | Capacity | Why |
|---|---|---|
| `s → (i, 0)` | ∞ | the first column is always on the source side |
| `(i, j) → (i, j+1)` | `cost[i, j]` | cutting this edge = removing pixel `(i, j)` |
| `(i, j+1) → (i, j)` | ∞ | if `(i, j+1)` is on the source side, so is `(i, j)` → the source side of each row is a **prefix**, so every row is cut **exactly once** |
| `(i, j) → (i-1, j-1)` and `(i, j) → (i+1, j-1)` | ∞ | the cut in neighbouring rows is at most one column apart → the seam is **connected** |
| `(i, W-1) → t` | `cost[i, W-1]` | see the note below |

**Note on the last column.** The assignment links the last column to the sink with ∞.
Then the last column is always on the sink side and can never be removed, while DP can
remove it. We give `(i, W-1) → t` the capacity `cost[i, W-1]` instead. This is exactly the
assignment's construction on the image padded with one extra column that is joined to the
sink with ∞, so it keeps the same proof but lets the seam use every column.

**Why the minimum cut is the best seam.** A cut is a split of the nodes into `S` (with `s`)
and `T` (with `t`); its capacity is the sum of the capacities of the edges going from `S` to `T`.

1. A finite cut never cuts an ∞ edge. The backward ∞ edges force row `i` of `S` to be a
   prefix `(i, 0) .. (i, p_i)`, and the diagonal ∞ edges force `|p_i − p_{i+1}| ≤ 1`.
   So `p` is a connected seam, and the only finite edge cut in row `i` is the one leaving
   `(i, p_i)`, with capacity `cost[i, p_i]`. **Cut capacity = seam cost.**
2. The other way round, every connected seam `p` gives a finite cut (take each row's prefix
   up to `p_i`). So cuts and seams match one to one.
3. Hence **minimum cut = minimum-cost seam**, and by the **max-flow min-cut theorem** the
   max-flow value equals that cost. The seam is optimal, just like the DP seam.

**Getting the seam from the flow.** After max-flow, BFS from `s` in the residual graph (only
arcs with residual capacity > 0). The reached nodes are the source side `S` of a minimum cut.
In each row the **rightmost pixel in `S`** is the seam pixel to remove.

---

## `src/algorithms/graphcut.py`

### `FlowNetwork` (a generic max-flow solver)
- Stores the **residual graph** in CSR form: the arcs leaving node `u` are
  `first[u] .. first[u+1]-1`; `head[a]` is where arc `a` points, `cap[a]` its residual
  capacity and `rev[a]` the index of its reverse arc.
- Every edge `u → v` is two arcs: forward (`cap`) and reverse (`rev_cap`, normally 0). Pushing
  `f` along an arc does `cap[a] -= f; cap[rev[a]] += f`.
- An edge with an opposite ∞ edge, like `(i, j) → (i, j+1)` with `cost` and `(i, j+1) → (i, j)`
  with ∞, is stored as **one arc pair** with `cap = cost`, `rev_cap = ∞`. This models the same
  residual network with fewer arcs.
- Arrays are Python lists, because indexing a list one item at a time is much faster than
  indexing a numpy array.
- `FlowNetwork.from_edges(n, [(u, v, c), ...])` builds a network from an edge list (used in the
  tests with the CLRS example, max-flow 23).

### `FlowNetwork.max_flow(source, sink, method="dinic") -> (flow, stats)`
- **`edmonds_karp`**: Ford–Fulkerson where each augmenting path is a **shortest** path, found by
  BFS. Push the bottleneck (smallest residual capacity on the path) and repeat until the sink
  cannot be reached. At most `O(VE)` augmentations × `O(E)` per BFS = **`O(VE²)`**.
- **`dinic`** (default): BFS once to build the **level graph** (`level[v]` = BFS distance), then a
  DFS finds a **blocking flow**: every shortest augmenting path in the level graph, using a
  "current arc" pointer per node so no arc is tried twice, and dropping dead-end nodes. The
  s–t distance grows after every phase, so there are at most `V` phases: **`O(V²E)`**.
  Same answer as Edmonds–Karp, but one BFS serves hundreds of augmenting paths.
- `stats` has `bfs_runs` and `augmenting_paths` (and `phases` for Dinic).
- The bottleneck of every s–t path is finite: to reach `t` a path must use a `(i, W-1) → t`
  arc, which has finite capacity. So ∞ (`float("inf")`) is safe to use.
- Capacities are integers for 8-bit images (Sobel and forward energy of integer pixels), so
  the flow values are exact.

### `FlowNetwork.min_cut_source_side(source) -> bool array`
BFS from the source over arcs with residual capacity > 0, after `max_flow`. Returns which
nodes are on the source side `S` of the minimum cut.

### `compute_cost_map(image, energy="backward") -> float64 (H, W)`
- `backward`: Sobel `|Gx| + |Gy|` via `sobel_edge_map`, **the same energy map as DP**, so the two
  methods can be compared directly.
- `forward`: `|I[i, j+1] − I[i, j−1]|` on the grayscale image (edge pixels replicated). This is
  the new edge created between the left and right neighbours when `(i, j)` is removed.

### `build_seam_graph(cost) -> (network, source, sink)`
Builds the graph in the table above with numpy (no Python loop over pixels). It has
`V = HW + 2` nodes and about `3HW` arc pairs (`4HW` edges, since each `∞` backward edge shares
its pair with a forward edge).

### `find_graphcut_seam(cost, method="dinic", return_info=False)`
The core of Question 4: **build graph → max-flow → BFS for `S` → rightmost `S` pixel per row**.
With `return_info=True` it also returns `max_flow`, `nodes`, `edges`, `bfs_runs`,
`augmenting_paths`. The max-flow value always equals `seam_cost(cost, seam)`.

**Ties:** when several seams have the same minimum cost, BFS returns the *smallest* source side,
so in every row the seam is as far left as an optimal seam can be. DP breaks ties differently,
so on images with ties the two seams can differ in pixels while having the same cost.

### `find_graphcut_horizontal_seam(cost, ...)`
The vertical seam of `cost.T`, so `seam[j]` is the row removed from column `j`.

### `seam_cost(cost, seam) -> float`
`sum(cost[i, seam[i]])`. Lower = better seam.

### `GraphCutSeamCarver.run(image, num_seams=1, **kwargs) -> AlgorithmResult`
Called by `main.py -a graphcut` and the GUI. Repeats `num_seams` times:
**cost map → graph cut seam → remove seam**. Extra keyword arguments:

| Argument | Values |
|---|---|
| `direction` | `"vertical"` (default, narrower image) or `"horizontal"` (shorter image, done by transposing the image, carving, and transposing back) |
| `energy` | `"backward"` (default, same as DP) or `"forward"` |
| `method` | `"dinic"` (default) or `"edmonds_karp"` |
| `visualize`, `step_delay`, `show_all_seams`, `step_callback`, `keep_steps` | same as the DP version |

`metadata` has the same keys as DP (`seams`, `seams_original`, `all_seams_image`, `step_frames`)
plus `seam_energies` (cost of each removed seam), `max_flows` (always equal to `seam_energies`)
and `solver_stats` (graph size, BFS runs and augmenting paths for each seam).

---

## Complexity (one seam, `V = HW + 2`, `E ≈ 3HW`)

| | Time | Space |
|---|---|---|
| DP | `O(HW)` | `O(HW)` (cost table + backtrack) |
| Graph cut, Edmonds–Karp | `O(VE²) = O((HW)³)` worst case | `O(HW)` (graph + residual capacities) |
| Graph cut, Dinic | `O(V²E) = O((HW)³)` worst case | `O(HW)` |
| Graph cut, any Ford–Fulkerson with integer costs | also `O(E · F)` where `F` = max-flow = seam cost | |

Removing `k` seams multiplies each time by `k`, because the graph is rebuilt for every seam.
Building the graph and the final BFS are `O(HW)`; the max-flow dominates. In practice, every
augmenting path is at least `W` arcs long, and the number of Dinic phases grows with the image
width (see the measurements below).

---

## Results

Measured with `python compare_q4.py --scale 0.2 -n 10 20 --sweep 0.05 0.1 0.15 0.2 0.25` on
`Broadway_tower_edit.jpg` scaled to 286×194 (Python 3.10, OpenCV 5.0, Windows 11, one core).
Times change between machines, but the ratios are what matter.

### First seam (same Sobel energy map)

| Method | Seam cost | Max-flow | Time (s) | BFS runs | Augmenting paths | Pixels ≠ DP |
|---|---|---|---|---|---|---|
| DP | 2060 | – | 0.026 | – | – | – |
| Graph cut, Dinic | 2060 | 2060 | 7.9 | 105 | 958 | 0 |
| Graph cut, Edmonds–Karp | 2060 | 2060 | 36.5 | 959 | 958 | 0 |

Graph: 55,486 nodes and 165,688 arc pairs (≈ 3HW). First horizontal seam: DP 544, graph cut
544 = max-flow, 0 pixels differ. Overlays: `output/q4_vertical_overlay.png`,
`output/q4_horizontal_overlay.png` (yellow = both methods chose the same pixel).

### Removing N vertical seams

Total cost = Sobel energy of the removed seams, recomputed after each removal (lower is better).

| Algorithm | N | Runtime (s) | s / seam | Total cost | Seams identical to DP |
|---|---|---|---|---|---|
| DP | 10 | 0.11 | 0.011 | 21,392 | – |
| Graph cut | 10 | 92.6 | 9.3 | 21,392 | 10 / 10 |
| Graph cut, forward energy | 10 | 51.2 | 5.1 | 25,676 | 0 / 10 |
| DP | 20 | 0.14 | 0.007 | 45,546 | – |
| Graph cut | 20 | 207.9 | 10.4 | 45,546 | 20 / 20 |
| Graph cut, forward energy | 20 | 133.6 | 6.7 | 68,294 | 0 / 20 |

### Running time vs image size (one seam)

| Size | Pixels | DP (ms) | Graph cut, Dinic (s) | Graph cut / DP | BFS runs | Augmenting paths |
|---|---|---|---|---|---|---|
| 71×48 | 3,408 | 1.7 | 0.09 | 57× | 24 | 356 |
| 143×97 | 13,871 | 5.6 | 0.87 | 155× | 46 | 592 |
| 214×145 | 31,030 | 5.3 | 3.6 | 672× | 68 | 776 |
| 286×194 | 55,484 | 8.3 | 9.1 | 1,099× | 105 | 958 |
| 357×242 | 86,394 | 10.5 | 13.0 | 1,237× | 103 | 1,030 |

On the full 1428×968 image, 3 seams did not finish within 10 minutes.

### Discussion

- **Seams.** Graph cut and DP found seams of the same cost every time, and the max-flow value
  always equalled that cost (max-flow min-cut theorem). On this image there were no ties, so
  the seams are identical pixel for pixel, and `q4_dp_nN.png` and `q4_graphcut_nN.png` are
  identical images.
- **Visual quality (backward energy).** Exactly the same as DP. Most seams run down the right
  border of the sky (smooth, and the Sobel x-gradient is 0 on the image border), so the result
  looks close to cropping the right side. The tower and the person are untouched.
- **Visual quality (forward energy).** The seams spread through the sky between the clouds and
  through the grass instead of piling up on the border, so more space is kept on the right of
  the tower and the framing stays more balanced. Its Sobel total is higher only because it
  minimises a different cost (the new edges created by removal), so judge it on the images
  (`q4_graphcut_forward_n20.png`, `..._seams.png`).
- **Running time.** Graph cut is 57× to 1,200× slower than DP for one seam, and the gap grows
  with image size. DP does one pass over the `HW` pixels. Dinic needs a number of phases that grows
  with the image width (24 → 105), and each phase's BFS and DFS touch most of the `HW` nodes, so the time is about
  `phases × HW`. Profiling at 214×145 shows about half the time in BFS and half in DFS.
  Edmonds–Karp needs one full BFS per augmenting path (959 BFS runs instead of 105) and is
  4.6× slower than Dinic.
- **Memory.** DP keeps an `H×W` table. Graph cut keeps about `6HW` arcs, each with a head, a
  residual capacity and a reverse index, so it uses several times more memory (both `O(HW)`).
- **When graph cut is worth it.** For a single image, DP gives the same seam far faster. A cut
  is more general: it still works when the seam is not a simple top-to-bottom path, such as a
  seam surface through a video in space-time (Rubinstein, Shamir and Avidan, 2008), where DP no
  longer applies.

---

## `tests/test_graphcut.py`: tests

Run: `.\.venv\Scripts\python.exe -m tests.test_graphcut`

| Function | What it checks |
|---|---|
| `test_textbook_max_flow` | The solver alone on the CLRS example network: max-flow 23 = min cut 23, for both methods. |
| `test_hand_traced_example` | 3×3 example: seam `[1, 2, 2]`, cost 2 + 1 + 1 = 4 = max-flow. |
| `test_last_and_first_column_can_be_removed` | Seams on either border column are found (the last-column fix). |
| `test_degenerate_sizes` | Width 1, height 1 and all-zero cost maps. |
| `test_matches_brute_force_on_random_input` | 150 random maps up to 5×5: max-flow = seam cost = brute-force optimum, and the seam is valid, for both methods. |
| `test_every_seam_is_a_finite_cut` | Every possible seam of a 4×4 map, turned into a cut, has capacity equal to its seam cost (the one-to-one argument above). |
| `test_matches_dp_on_larger_input` | 30×40 random maps: graph cut seam cost = DP seam cost = max-flow. |
| `test_horizontal_seam` | Horizontal seam is valid and optimal. |
| `test_cost_maps` | Forward energy on a black/white edge, backward energy shape. |
| `test_run_vertical_horizontal_and_forward` | `run()` output shapes and metadata for every option. |
| `test_run_matches_dp_carver` | First seam of `GraphCutSeamCarver` and `DPSeamCarver` on an image have equal cost. |
| `test_invalid_arguments` | Bad `num_seams`, `direction`, `energy` or `method` raise `ValueError`. |

---

## `compare_q4.py`: the Q4 comparison (Graph Cut vs DP)

Run: `.\.venv\Scripts\python.exe compare_q4.py [image] [--scale 0.1] [-n 5 10 20] [--sweep 0.05 0.1 0.15 0.2]`

The max-flow is plain Python, so the image is scaled down first (`--scale`). DP runs on the same
scaled image, so the comparison is fair.

1. **First seam** on one shared Sobel energy map: seam cost, max-flow value, time, BFS runs,
   augmenting paths and number of pixels that differ from the DP seam, for DP, Dinic and
   Edmonds–Karp. Saves `output/q4_vertical_overlay.png` (green = DP, red = graph cut,
   yellow = same pixel).
2. **First horizontal seam**: graph cut vs DP on the transposed image →
   `output/q4_horizontal_overlay.png`.
3. **N seams** with DP, graph cut (backward energy) and graph cut (forward energy): runtime, total
   Sobel energy removed and how many seams are identical to DP's. Saves
   `output/q4_{dp,graphcut,graphcut_forward}_n{N}.png` and `..._seams.png` (all seams overlaid).
4. `--sweep`: running time of one seam vs image size.

---

## Key points for the quiz
- **Seam → cut:** ∞ backward edges = one cut per row; ∞ diagonal edges = connected seam; the
  finite edge `(i, j) → (i, j+1)` carries the cost of removing `(i, j)`.
- **Max-flow min-cut theorem:** max-flow value = min cut capacity = minimum seam cost, so graph
  cut is **optimal, exactly like DP** (same seam cost every time; identical seams unless there
  are ties).
- **Seam from the flow:** BFS from `s` in the residual graph → `S`; rightmost `S` pixel per row.
- **Edmonds–Karp** = BFS shortest augmenting paths, `O(VE²)`. **Dinic** = level graph +
  blocking flow, `O(V²E)`, far fewer BFS runs in practice.
- **Last column:** capacity `cost[i, W−1]` to the sink instead of ∞ so it can be removed.
- **Horizontal seams:** transpose, find a vertical seam, transpose back.
- **Speed:** DP is `O(HW)` per seam and takes milliseconds; graph cut is orders of magnitude
  slower, and the gap grows with image size. Graph cut is worth it only when the problem is not
  a simple path, e.g. seams through video (a surface in space-time) or extra constraints, where
  DP no longer works but a cut still does.
