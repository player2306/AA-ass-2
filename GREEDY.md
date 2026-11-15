# Greedy Seam Carving (Question 3), by Member 3 (Edwin)

This file covers every function written for the Greedy part, what it does, and how to run it.

A **vertical seam** is stored as a 1-D int array `seam` of length `H` (image height).
`seam[r]` is the column of the pixel removed from row `r`.

---

## `src/seam_utils.py`: shared helpers (Greedy and DP can both use these)

### `compute_energy(image) -> np.ndarray`
- **Input:** `image`, the BGR image as a `uint8` array of shape `(H, W, 3)`.
- **Output:** a `float64` energy map of shape `(H, W)`.
- **What it does:** converts the image to grayscale and applies the Sobel filter in x and y. Energy = `|dx| + |dy|`.
  High energy = edges and detail (keep them). Low energy = flat areas (safe to remove).
- **Note:** it uses `BORDER_REPLICATE`. With OpenCV's default (reflect) border, the gradient on the left/right edge is always 0, so every seam would run down the image border and the result would just be a crop.

### `remove_vertical_seam(image, seam) -> np.ndarray`
- **Input:** the image `(H, W, 3)` and a seam (length `H`).
- **Output:** a new image `(H, W-1, 3)`.
- **What it does:** builds a True/False mask that is False at `(r, seam[r])` in each row, keeps the True pixels, and reshapes the result to one column narrower.

### `seam_energy(energy, seam) -> float`
- **What it does:** returns the total energy of the seam, `sum(energy[r, seam[r]])`.
  **Lower = better seam.** Q3 uses this number to compare seam quality between Greedy and DP.

### `draw_seam(image, seam, color=(0, 0, 255)) -> np.ndarray`
- **What it does:** returns a copy of the image with the seam pixels painted in `color` (BGR, so the default is red).
  It is used for the seam-overlay image in the Q3 comparison.

---

## `src/algorithms/greedy.py`: the Greedy algorithm

### `find_greedy_seam(energy) -> np.ndarray`
The core of the Greedy algorithm.
- **Input:** an energy map `(H, W)`.
- **Output:** a seam (int array of length `H`).
- **Steps:**
  1. `seam[0]` = column with the lowest energy in the top row.
  2. For each next row, look only at the (up to) 3 pixels below the previous one: down-left, down and down-right. Pick the cheapest.
  3. Never look further ahead and never undo a choice.
- **Boundaries:** the window is `left = max(prev-1, 0)` to `right = min(prev+1, W-1)`. On an edge column there are only 2 candidates; the seam never wraps around.
- **Ties:** `np.argmin` returns the *first* minimum, so the leftmost candidate wins. This makes the result deterministic.
- **Detail:** `left + np.argmin(energy[r, left:right+1])`. `argmin` gives the position inside the slice, so `left` is added back to get the real column.
- **Complexity:** O(H + W) time per seam (3 comparisons per row, plus the top row). Extra space O(H).

```
seam[0] = argmin(energy[0])
for r = 1 .. H-1:
    p = seam[r-1]
    seam[r] = cheapest column among p-1, p, p+1 (kept inside 0..W-1)
```

### `GreedySeamCarver.run(image, num_seams=1) -> AlgorithmResult`
The method `main.py` calls (`-a greedy`).
- **Input:** the original image and the number of vertical seams to remove.
- **Output:** `AlgorithmResult` containing:
  - `image`: the carved image, `(H, W - num_seams, 3)`
  - `algorithm`: `"greedy"`
  - `runtime_s`: wall-clock time, measured with `timer()` from `src/utils/timing.py`
  - `metadata["seam_energies"]`: the total energy of each removed seam
- **What it does:** repeats `num_seams` times: **compute energy → find greedy seam → remove it.**
  The energy is recomputed every time because removing a seam makes new pixels neighbours, which changes their gradients.
- Raises `ValueError` if `num_seams` is negative or would remove the whole image.
- **Complexity for k seams:** O(k · H · W). The energy recomputation dominates, which is the same order as DP; Greedy only has a smaller constant.

---

## `tests/test_greedy.py`: tests

Run: `.\.venv\Scripts\python.exe -m tests.test_greedy`

| Function | What it checks |
|---|---|
| `best_seam_energy(energy)` | Helper, not a test. Brute-forces **every** valid seam and returns the lowest total energy. This is the true optimum, which DP must also find. Exponential time, so tiny matrices only. |
| `test_hand_traced_example` | 3×3 example traced by hand: the greedy seam is `[1, 0, 0]`. |
| `test_greedy_worse_than_optimal` | **Counterexample:** Greedy's seam costs **102** while the optimal one costs **4**. Greedy takes the cheap 1s on the left and then gets trapped above the 100s. This proves Greedy is not optimal. |
| `test_seam_is_valid_on_random_input` | The seam has length H, stays inside the image, and is connected (neighbouring rows differ by at most 1 column). |
| `test_greedy_never_beats_optimal` | Across 200 random 5×4 matrices, Greedy's seam energy is never below the optimum. |
| `test_edges_and_ties` | Width-1 image, all-equal energies (the leftmost column wins), and a seam on the right edge (no out-of-bounds access). |
| `test_remove_seam_and_run` | `remove_vertical_seam` drops the right pixel, and `run()` returns the right shape and one energy per seam. |

---

## `compare_q3.py`: the Q3 comparison (Greedy vs DP)

Run: `.\.venv\Scripts\python.exe compare_q3.py [image] [-n 10 50 100]`

### `main()`
1. Computes **one energy map** and finds the first seam with each method on that same map, so the comparison is fair. Prints each seam's total energy.
2. Saves `output/q3_seam_overlay.png`: the Greedy seam in **red** and the DP seam in **green**.
3. For each N in `-n`, runs both algorithms, prints a table of **runtime** and **total seam energy**, and saves `output/q3_greedy_nN.png` and `output/q3_dp_nN.png` for the visual comparison.

The DP rows only appear once Member 1 adds `find_dp_seam(energy)` and `DPSeamCarver.run` to `dp.py`. Until then the script prints "not available" / "not implemented" for DP.

---

## Key points for the quiz
- **Greedy** makes a local, row-by-row choice with no lookahead. It is fast but not optimal.
- **DP** considers all seams through its cumulative table, so it always finds the minimum-energy seam.
- **Counterexample:** Greedy 102 vs optimal 4.
- **Finding one seam:** Greedy O(H + W) time / O(H) space vs DP O(H·W) time / O(H·W) space.
- **Removing k seams:** both are O(k·H·W), because the energy is recomputed after every removal.
- **Tie-break:** leftmost. **Boundaries:** clamped, never wrapped around.
