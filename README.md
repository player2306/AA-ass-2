# AA ass 2 — Seam Carving

Base OpenCV project for the seam carving assignment. Image loading, preview,
saving, timing, and the algorithm interface are ready. The algorithms themselves
are stubs so each member can add their own implementation.

## Setup

Requires Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, only run the install command when dependencies change.

## Run

Run from the project root:

```powershell
# Preview the image without running an algorithm
.\.venv\Scripts\python.exe main.py --preview

# Run an algorithm
.\.venv\Scripts\python.exe main.py -a dp -n 1
.\.venv\Scripts\python.exe main.py Broadway_tower_edit.jpg -a greedy -n 50
.\.venv\Scripts\python.exe main.py -a graphcut -n 100 -o output\graphcut.jpg --no-show
```

CLI options:

| Option | Description |
| --- | --- |
| `image` | Input image path (default: `Broadway_tower_edit.jpg`) |
| `-a`, `--algorithm` | `dp`, `greedy`, or `graphcut` (default: `dp`) |
| `-n`, `--seams` | Number of vertical seams to remove (default: `1`) |
| `-o`, `--output` | Output path (default: `output/result.jpg`) |
| `--preview` | Load and display only, no algorithm |
| `--no-show` | Skip the preview window |

Algorithms currently print `not implemented` until their module is filled in.

## Structure

```text
main.py                     CLI entry point
src/config.py               Paths and display constants
src/image_io.py             OpenCV load/save helpers
src/display.py              Preview window helpers
src/utils/timing.py         Wall-clock timer for benchmarks
src/algorithms/base.py      SeamCarver interface + AlgorithmResult
src/algorithms/dp.py        Member 1 — DP core engine (stub)
src/algorithms/greedy.py    Member 3 — greedy implementation (stub)
src/algorithms/graphcut.py  Member 4 — graph cut / max-flow (stub)
output/                     Saved results
```

## Adding an algorithm

1. Implement `run` in the relevant module under `src/algorithms/`.
2. Return `AlgorithmResult(image=..., algorithm=..., runtime_s=..., metadata=...)`.
3. If you add a new module, register its class in `src/algorithms/__init__.py`.

Input and output images are BGR `numpy.uint8` arrays, matching OpenCV.
