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

## Troubleshooting

### `.venv : The term '.venv' is not recognized`

You cannot activate a venv by typing its folder name. Run the activation script
with the full path:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or skip activation entirely and call the interpreter directly (what the Run
commands below do):

```powershell
.\.venv\Scripts\python.exe main.py --gui
```

### `Activate.ps1 cannot be loaded because running scripts is disabled`

PowerShell's execution policy blocks the activation script. Allow it for the
current shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `python` opens the Microsoft Store or is not recognized

On Windows, `python` is often an App Execution Alias that launches the Store
instead of a real interpreter. Use the `py` launcher to check what is available:

```powershell
py --version
py -0p        # list installed interpreters and their paths
```

Create the venv with an explicit version, e.g. `py -3.11 -m venv .venv`.

### `pip install` fails to find a wheel (especially `opencv-python`)

Very new Python releases may not have prebuilt wheels yet. Confirm the venv is
using the interpreter you expect and, if needed, recreate it on a supported
version:

```powershell
.\.venv\Scripts\python.exe --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### GUI crashes with `TclError: Can't find a usable init.tcl`

```
_tkinter.TclError: Can't find a usable init.tcl in the following directories: ...
This probably means that Tcl wasn't installed properly.
```

Tcl is installed fine. This is a known bug in **Python 3.13.0**, where tkinter
looks in the wrong folders for its Tcl files when run from a venv. Check which
version your venv uses:

```powershell
.\.venv\Scripts\python.exe --version
```

If it says `3.13.0`, either upgrade to Python 3.13.1 or later, or rebuild the
venv on another installed version (3.11, 3.12 or 3.14 all work):

```powershell
Remove-Item -Recurse -Force .venv
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Verifying the environment

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
.\.venv\Scripts\python.exe -c "import tkinter; tkinter.Tk().destroy(); print('tkinter OK')"
```

If both commands succeed, the environment is set up correctly and the GUI will
start.

## Run

Run from the project root:

```powershell
# Simple GUI (recommended for interactive size editing + visualization)
.\.venv\Scripts\python.exe main.py --gui
# or
.\.venv\Scripts\python.exe gui.py

# Preview the image without running an algorithm
.\.venv\Scripts\python.exe main.py --preview

# Run an algorithm
.\.venv\Scripts\python.exe main.py -a dp -n 1
.\.venv\Scripts\python.exe main.py Broadway_tower_edit.jpg -a greedy -n 50
.\.venv\Scripts\python.exe main.py -a graphcut -n 100 -o output\graphcut.jpg --no-show

# Carve until the image is exactly 1200x900 (or give only --width / --height)
.\.venv\Scripts\python.exe main.py -a dp --width 1200 --height 900

# Step-by-step visualization and all-seams overlay
.\.venv\Scripts\python.exe main.py -a dp -n 20 --visualize --show-all-seams -o output\dp.jpg

# Keyboard interactive mode (OpenCV window)
.\.venv\Scripts\python.exe main.py -a dp -i --step-size 5 -o output\interactive.jpg
```

CLI options:

| Option | Description |
| --- | --- |
| `image` | Input image path (default: `Broadway_tower_edit.jpg`) |
| `-a`, `--algorithm` | `dp`, `greedy`, or `graphcut` (default: `dp`) |
| `-n`, `--seams` | Number of seams to remove (default: `1`) |
| `-d`, `--direction` | `vertical` (narrower), `horizontal` (shorter) or `both` (default: `vertical`) |
| `--width`, `--height` | Carve until the image is this size (replaces `-n`/`-d`; either one alone keeps the other side). Can only shrink |
| `-o`, `--output` | Output path (default: `output/result.jpg`) |
| `--preview` | Load and display only, no algorithm |
| `--no-show` | Skip the preview window |
| `--visualize` | Step-by-step seam removal preview (OpenCV window) |
| `--step-delay` | Delay (ms) between visualization steps (`0` waits for key) |
| `--show-all-seams` | Overlay all removed seams on the original/base image |
| `--all-seams-output` | Save path for the all-seams overlay image |
| `-i`, `--interactive` | Keyboard interactive resizing mode (OpenCV window) |
| `--step-size` | Seams per interactive step |
| `--gui` | Launch simple Tkinter GUI |

All three algorithms (`dp`, `greedy`, `graphcut`) are implemented.

### GUI

`main.py --gui` (or `gui.py`) opens one window with every setting:

| Control | Same as |
| --- | --- |
| Open… / Save… / Save seams overlay… | `image`, `-o`, `--all-seams-output` |
| Algorithm, Direction, Seams, **Remove seams** | `-a`, `-d`, `-n` |
| Target size W × H, **Carve to size** (or Enter in a field) | `--width`, `--height` |
| Step-by-step + Delay ms | `--visualize`, `--step-delay` |
| Show all seams | `--show-all-seams` |
| Multi-start greedy (greedy only) | `multi_start=True` |
| Energy, Max-flow (graph cut only) | `energy=`, `method=` |
| Scale % + Apply scale | shrink the image first (`compare_q4.py --scale`) |
| Undo, Reset, Before / after | — |

Keyboard shortcuts (as in `-i` mode): `A`/Left narrower, `W`/Up shorter, `B` both,
`D`/Right reset, `S` save, `Ctrl+Z` undo.

Before a graph cut run that is estimated to take more than about 5 seconds, the GUI
shows a warning with the rough time and a suggested `Scale %`, and asks whether to
continue (default: No). Estimates come from `SLOW_ALGORITHMS` in `gui.py`.

`graphcut` solves a max-flow problem in plain Python for every seam, so it is far slower
than `dp`: about 1 s per seam on a 143x97 image and about 9 s on 286x194, while on the
full-size image a few seams take over 10 minutes. Use a small image, or
`compare_q4.py --scale`, when trying it.

## Question 4: Graph cut (min-cut / max-flow)

Details, proofs and results are in [GRAPHCUT.md](GRAPHCUT.md).

```powershell
# Tests (brute force, DP and textbook max-flow checks)
.\.venv\Scripts\python.exe -m tests.test_graphcut

# Graph cut vs DP: seams, visual quality, running time (writes output\q4_*.png)
.\.venv\Scripts\python.exe compare_q4.py --scale 0.1 -n 5 10 20 --sweep 0.05 0.1 0.15 0.2
```

## Tests

Install the test dependency once, then run the whole suite with one command
(about 60 tests, a couple of seconds; no OpenCV windows are opened):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

| File | Checks |
| --- | --- |
| `tests/test_core.py` | DP finds the optimal seam (brute force) and its fast energy update is exact; every algorithm keeps original pixels in order, rejects bad arguments, handles 1-pixel and grayscale images; carving to a size; image I/O errors |
| `tests/test_cli.py` | `main.py` end to end: output sizes, error exit codes |
| `tests/test_gui.py` | GUI actions: remove, carve to size, undo/reset, warnings, saving |
| `tests/test_greedy.py`, `tests/test_graphcut.py` | The algorithm authors' own correctness tests |

Add `-v` to list every test, or a file path to run one file.

## Structure

```text
main.py                     CLI entry point
src/config.py               Paths and display constants
src/image_io.py             OpenCV load/save helpers
src/display.py              Preview window helpers
src/utils/timing.py         Wall-clock timer for benchmarks
src/algorithms/base.py      SeamCarver interface + AlgorithmResult
src/algorithms/dp.py        Member 1 — DP core engine
src/algorithms/greedy.py    Member 3 — greedy implementation (see GREEDY.md)
src/algorithms/graphcut.py  Member 4 — graph cut / max-flow (see GRAPHCUT.md)
compare_q4.py               Question 4 comparison: graph cut vs DP
tests/                      Test suite (see Tests above)
output/                     Saved results
```

## Adding an algorithm

1. Implement `run` in the relevant module under `src/algorithms/`.
2. Return `AlgorithmResult(image=..., algorithm=..., runtime_s=..., metadata=...)`.
3. If you add a new module, register its class in `src/algorithms/__init__.py`.

Input and output images are BGR `numpy.uint8` arrays, matching OpenCV.
