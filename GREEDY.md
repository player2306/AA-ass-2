# Greedy Seam Carving (Question 3), by Member 3 (Edwin)

This file covers every function written for the Greedy part, what it does, how
to run it, and the results of the Greedy vs DP comparison.

A **vertical seam** is stored as a 1-D int array `seam` of length `H` (image height).
`seam[r]` is the column of the pixel removed from row `r`. Neighbouring rows differ
by at most 1 column, so the seam is connected.

**Files for this part:** `src/algorithms/greedy.py`, `src/seam_utils.py`,
`compare_q3.py`, `tests/test_greedy.py`.

---

## 1. How Greedy works

```
seam[0] = argmin(energy[0])                 # cheapest pixel in the top row
for r = 1 .. H-1:
    p = seam[r-1]
    seam[r] = cheapest column among p-1, p, p+1   (kept inside 0..W-1)
```

- It is a **local, row-by-row decision**. It only ever compares at most 3 numbers,
  never looks further ahead, and never changes a choice. Nothing is sorted.
- **Boundaries:** the window is `max(p-1, 0)` to `min(p+1, W-1)`, so a seam on the
  left/right edge only has 2 candidates. It never wraps around.
- **Tie-break:** `np.argmin` returns the **first** minimum, so the **leftmost**
  candidate wins (in the top row and in every step). The result is deterministic.
- **Greedy is not guaranteed to find the optimal seam.** It can commit to a cheap
  pixel that leads into an expensive area. Counterexample from the tests:

  ```
  energy = [[  1,   2, 2],
            [  1,   5, 1],
            [100, 100, 1]]
  ```
  Greedy takes column 0 all the way down: 1 + 1 + 100 = **102**.
  The optimal seam is columns [1 or 2, 2, 2]: 2 + 1 + 1 = **4**.

### How DP differs (for comparison)
DP (Member 1) builds a table `M` where `M[r][c]` is the **minimum cumulative
energy of any seam from the top row down to pixel (r, c)**:

```
M[0][c] = e[0][c]
M[r][c] = e[r][c] + min(M[r-1][c-1], M[r-1][c], M[r-1][c+1])
```

Each entry **reuses the already-solved results of the row above** (overlapping
subproblems), so DP does **not** enumerate seams, even though the number of
possible seams grows exponentially. The optimal seam ends at the smallest value
in the bottom row and is found by backtracking. Greedy uses the same
"3 pixels below" rule, but on the raw energy instead of these cumulative costs.

---

## 2. Complexity

| | Time | Space |
|---|---|---|
| Greedy, one seam (`find_greedy_seam`) | **O(W + H)**: O(W) to scan the top row, then O(1) per row | **O(H)** for the returned seam, **O(1)** extra working space |
| Multi-start greedy, one seam | O(H · W): W walks, each H steps | O(H · W) to store all W paths |
| DP, one seam | O(H · W): every pixel is filled once | O(H · W) for the cost and backtrack tables |
| Energy map (Sobel), or removing one seam | O(H · W) | O(H · W) |

**Removing k seams.** Each seam needs a new energy map and a removal, both
proportional to the **current** image size. The i-th seam (i = 0 .. k-1) is
removed from an image of width W − i, so the total work is

```
H·W + H·(W-1) + ... + H·(W-k+1)  =  H · (kW − k(k−1)/2)
```

Since k < W, k(k−1)/2 < kW/2, so this lies between ½·kHW and kHW:
the total is **Θ(k·H·W)**, a tight bound (not just an upper bound).
Greedy's seam search adds only O(W + H) per seam, so Greedy is Θ(kHW) overall.
DP's seam search is also Θ(HW) per seam, so DP is Θ(kHW) too. Both are the same
order; Greedy only has a smaller constant.

---

## 3. Energy map and the border

The energy is the team's shared `sobel_edge_map` (`src/utils/filters.py`):
grayscale, Sobel 3×3, energy = |Gx| + |Gy|, `float32`, shape (H, W).

**Border mode.** Sobel needs the pixel to the left and right of each pixel. On the
image edge one of them does not exist:
- **Default (`BORDER_REFLECT_101`)** mirrors the image, so the missing left pixel
  becomes a copy of the right one. That makes **Gx exactly 0 in the first and last
  columns**: edge pixels get only |Gy|, so they look cheap and seams are biased
  towards the image edges. Measured: with the default border, 11 of 50 Greedy
  seams ran along the edge and the first Greedy seam was in columns 0–25.
- **`BORDER_REPLICATE`** (now used) copies the edge pixel, so the real change in
  brightness is still detected. This **greatly reduces the bias, but a small bias
  remains**: the edge gradient spans 1 pixel instead of 2. After the change,
  0 of 50 Greedy seams ran along the edge. DP's seams did not change.
- The border is set once (`SOBEL_BORDER` in `filters.py`) and also used by DP's
  incremental energy update, so the two always agree (checked: difference 0).

---

## 4. Functions

### `src/algorithms/greedy.py`

#### `find_greedy_seam(energy, start=None) -> np.ndarray`
The core Greedy algorithm (section 1).
- **Input:** energy map `(H, W)`; optional `start` column. By default it starts at
  the cheapest top-row pixel.
- **Output:** seam, an int array of length `H`.
- **Detail:** `left + np.argmin(energy[r, left:right+1])`. `argmin` gives the
  position inside the slice, so `left` is added back to get the real column.

#### `find_multistart_greedy_seam(energy) -> np.ndarray`
**Alternative design.** Runs Greedy from **every** top-row column and keeps the
seam with the lowest total energy. This fixes the "bad starting point" problem,
but every walk is still greedy after its start, so it is **still not guaranteed
optimal**.
- All W walks run at the same time with NumPy: `cols` holds every walk's current
  column, and each row moves all of them one step, using the same
  boundary and tie-break rules as `find_greedy_seam`.
- O(H·W) time, the same order as DP, but with worse seams (section 6). Once you
  pay O(H·W), DP is the better choice because it is guaranteed optimal.

#### `GreedySeamCarver.run(image, num_seams=1, **kwargs) -> AlgorithmResult`
Called by `main.py` (`-a greedy`) and by the GUI.
- Repeats `num_seams` times: **compute energy → find seam → remove it**. The energy
  is recomputed because removing a seam gives pixels new neighbours, which changes
  their gradients.
- Raises `ValueError` unless `0 <= num_seams < width`.
- **Options** (the same ones DP accepts):
  - `step_callback`: the GUI's step-by-step view;
  - `visualize`, `step_delay`: `main.py --visualize`;
  - `show_all_seams`: the GUI's "Show All Seams";
  - `multi_start=True`: use the multi-start variant.
  Frames are only drawn when one of these asks for them, so normal runs and the
  Q3 timings don't pay for drawing.
- **Returns** `AlgorithmResult` with:
  - `image`: the carved image `(H, W − num_seams, 3)`;
  - `algorithm`: `"greedy"` or `"greedy_multi"`;
  - `runtime_s`: total wall-clock time;
  - `metadata`:
    - `seam_energies`: the total energy of each removed seam;
    - `seams`: each seam, in the image it was removed from;
    - `seams_original`: the same seams in original-image columns (tracked with a
      `col_map` that has the same seams removed);
    - `all_seams_image`: all seams drawn on the original image, if requested;
    - `timing`: `energy_s`, `seam_s`, `removal_s`, `other_s`, a breakdown of
      `runtime_s` (other = bookkeeping, column tracking, drawing).

#### `_make_frame(image, seam, step, total)`
Helper for the GUI and `--visualize`: the current image with the seam in red and
a "Seam i/N" label. It is not part of the algorithm.

### `src/seam_utils.py`
A small compatibility layer. Energy, removal and drawing now come from the
shared `src/utils` code, so Greedy and DP use **exactly the same functions**.
- `compute_energy(image)`: calls the shared `sobel_edge_map`.
- `remove_vertical_seam`, `draw_seam`: re-exported from `src/utils`.
- `seam_energy(energy, seam) -> float`: total energy along a seam, summed in
  float64 for accuracy. **Lower = better seam.** The shared code has no
  equivalent.

---

## 5. Tests: `tests/test_greedy.py`

Run: `.\.venv\Scripts\python.exe -m tests.test_greedy` (9 tests).

| Test | What it checks |
|---|---|
| `best_seam_energy` (helper) | Brute-forces **every** valid seam and returns the lowest total. This is the true optimum. Exponential time, so tiny matrices only. |
| `test_hand_traced_example` | 3×3 example traced by hand gives the seam `[1, 0, 0]`. |
| `test_greedy_worse_than_optimal` | The counterexample: Greedy 102 vs optimal 4. |
| `test_seam_is_valid_on_random_input` | Length H, inside the image, connected. |
| `test_greedy_never_beats_optimal` | 200 random matrices: Greedy ≥ optimum. |
| `test_edges_and_ties` | Width 1, all-equal energies (leftmost wins), seam on the right edge. |
| `test_remove_seam_and_run` | Seam removal, output shape of `run()`, and the timing breakdown adds up to the total. |
| `test_multistart_greedy` | 200 random matrices: multi-start = best of every single start, and optimum ≤ multi-start ≤ greedy. Finds 4 on the counterexample. |
| `test_dp_matches_brute_force` | DP's seam = brute-force optimum on 300 random matrices, and Greedy is never better than DP. Skipped if DP is missing. |
| `test_seam_similarity` | The similarity helper used by `compare_q3.py`. |

---

## 6. The Q3 comparison: `compare_q3.py`

Run: `.\.venv\Scripts\python.exe compare_q3.py [image] [-n 10 50 100] [-r 5] [-s 0.25 0.5 1 2]`
(`-n` seams to remove, `-r` repeats per measurement, `-s` image scales).

Every time is the **median of r repeats**. In section 3 the order of the algorithms
alternates on each repeat, so none always runs first. Invalid `-n` values are
skipped, and a missing or unregistered DP doesn't crash the script.

| Section | What it measures | Output |
|---|---|---|
| 1. First seam | Greedy, multi-start and DP seams on the **same** energy map: total energy, Greedy/DP ratio, % of rows where Greedy and DP match, mean and max column distance | `q3_first_seam.csv`, `q3_seam_overlay.png` (red = greedy, green = DP, yellow = both), `q3_seam_overlay_thick.png` (drawn 5 px wide for the report) |
| 2. One seam, step by step | ms for energy, seam finding and removal | `q3_step_timing.csv` |
| 3. Removing N seams | `run()` runtime (median, min, max) and total seam energy | `q3_runs.csv`, `q3_{alg}_n{N}.png`, `q3_{alg}_all_seams_n{N}.png` |
| 4. Image sizes | step times at several scales, divided by the size Big-O predicts | `q3_sizes.csv` |

DP doesn't record its seam energies, so the script replays DP's stored seams to
work them out, outside the timed run.

### Results: Broadway tower, 1428×968, median of 5
Timings vary between runs on the same machine, so quote medians and ranges.

**First seam (same energy map):** Greedy 23,806 · multi-start 14,974 · DP **11,266**.
Greedy's seam carries 2.11× DP's energy. The two seams share **0%** of rows and are
about 1,129 px apart on average: completely different paths.

**One seam, step by step (ms):**

| | energy | find seam | remove |
|---|---|---|---|
| Greedy | 7.9 | **1.1** | 3.2 |
| Multi-start | 7.9 | 39.4 | 3.2 |
| DP | 7.9 | 41.2 | 3.2 |

**Removing N seams:**

| N | Greedy | Multi-start | DP | Greedy energy | Multi-start energy | DP energy |
|---|---|---|---|---|---|---|
| 10 | 0.17 s | 0.56 s | 0.60 s | 356,674 | 152,034 | **119,074** |
| 50 | 0.85 s | 2.74 s | 2.88 s | 1,404,162 | 787,606 | **658,764** |
| 100 | 1.70 s | 5.68 s | 5.98 s | 2,499,398 | 1,575,914 | **1,426,046** |

**Image sizes:** Greedy's seam time per row stays about 1–1.3 µs from 357×242 up
to 2856×1936, so its search is linear in H, as predicted. DP's time per pixel falls
from about 114 to 24 ns, because its Python loop has a fixed cost per row: its
cost behaves like a·H + b·H·W. That is still O(H·W); the fixed per-row part just
matters most for small images.

### What the results show
1. **Speed:** Greedy finds a seam about **40× faster** than DP (O(W + H) vs O(H·W)),
   but is only about **3.5× faster overall**. Energy recomputation and removal cost
   the same for both and take most of the time (43% and 20% of a Greedy run).
2. **Seam quality:** Greedy's seams carry about **1.8–2.1× more energy** than DP's.
3. **Visual quality:** after 100 seams, DP's seams stay in the empty sky to the
   right of the tower and leave the tower and the person untouched. Greedy's seams
   spread across the sky, and several **cut straight through the tower** (see the
   `q3_*_all_seams_n100.png` images). They start in the cheap sky above the tower,
   and with no lookahead they cannot avoid the stone below.
   **Conclusion: Greedy is not comparable to DP in visual quality.**
4. **Multi-start greedy** costs as much as DP but is still worse, so DP is the
   better use of O(H·W) time.

### Caveats
- **Totals over N seams are not a strict like-for-like comparison.** After the
  first seam, Greedy and DP have removed different pixels, so they continue on
  different images. Only section 1 compares the two on exactly the same input.
- DP's `run()` updates its energy **incrementally** after each seam, while Greedy
  recomputes the whole map. This slightly favours DP in the total runtimes.
  Section 2 times each step on the same input, which is the fair comparison.
- The all-seams view shows Greedy seams **sweeping** across the image. Each new
  seam starts at the cheapest top-row pixel, ties go to the leftmost one, and each
  removed seam uses up its starting pixel.

---

## 7. Horizontal seams

Not implemented yet for either algorithm. DP (Member 1) currently does vertical
seams only, and Q1 requires both. A horizontal seam is a vertical seam of the
image with rows and columns swapped, so either algorithm can support it by
transposing (`np.transpose(img, (1, 0, 2))`), running the vertical code, and
transposing back. No new algorithm is needed.

---

## 8. Key points for the quiz
- Greedy = local, row-by-row choice among at most 3 pixels below. It does **no
  sorting** and has **no lookahead**. It is **not guaranteed optimal**
  (counterexample: 102 vs 4).
- DP stores the minimum cumulative energy to reach each pixel and **reuses** the
  row above's results. It doesn't enumerate seams, and it always finds the
  optimal seam.
- One seam: Greedy O(W + H) time, O(H) space (O(1) extra); DP O(H·W) time and
  space.
- k seams: H(kW − k(k−1)/2) = **Θ(kHW)** for both, because the energy is
  recomputed after every removal.
- Tie-break: leftmost (`np.argmin`). Boundaries: clamped, never wrapped.
- Border: the default makes Gx = 0 on the edge columns, which pulls seams to the
  edges; REPLICATE greatly reduces this, with a small bias left.
- Multi-start greedy is O(H·W), like DP, and still worse than DP.
- Results: Greedy is about 40× faster at finding a seam, about 3.5× faster
  overall, has about 2× the seam energy, and cuts through the tower. **Not
  comparable in visual quality.**
