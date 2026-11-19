# AA ass 2 — Seam Carving

OpenCV project for the seam carving assignment. It includes dynamic programming,
greedy and graph-cut implementations, image loading, preview, saving and timing.

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
.\.venv\Scripts\python.exe main.py -a graphcut -n 1 -o output\graphcut.jpg --no-show

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
| `-n`, `--seams` | Number of seams to remove per selected direction (default: `1`) |
| `-d`, `--direction` | Remove `vertical`, `horizontal`, or `both` directions (default: `vertical`) |
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

Graph cut solves a max-flow problem for every seam and can be much slower than
DP, especially on large images. Use a small image, or `compare_q4.py --scale`,
when trying it. Results are generated locally in `output/`.

## Tests and benchmarks

```powershell
# Greedy, DP and graph-cut checks
.\.venv\Scripts\python.exe -m tests.test_greedy
.\.venv\Scripts\python.exe -m tests.test_graphcut
.\.venv\Scripts\python.exe -m tests.test_dp_optimization

# Reproduce the Question 5 comparisons and DP memory measurement
.\.venv\Scripts\python.exe benchmark_q5.py --repeats 3
.\.venv\Scripts\python.exe benchmark_q5_optimization.py --repeats 7

# Optional graph-cut and DP comparison on a scaled image
.\.venv\Scripts\python.exe compare_q4.py --scale 0.05 -n 1 3 5
```

The Question 5 benchmarks write CSV files and graphs to `output/q5_benchmark/`.

## Structure

```text
main.py                     CLI entry point
src/config.py               Paths and display constants
src/image_io.py             OpenCV load/save helpers
src/display.py              Preview window helpers
src/utils/timing.py         Wall-clock timer for benchmarks
src/algorithms/base.py      SeamCarver interface + AlgorithmResult
src/algorithms/dp.py        DP core engine
src/algorithms/greedy.py    Greedy implementation
src/algorithms/graphcut.py  Graph cut / max-flow
compare_q3.py               Greedy vs DP comparison
compare_q4.py               Question 4 comparison: graph cut vs DP
benchmark_q5.py             Runtime comparison for all three algorithms
benchmark_q5_optimization.py  DP seam-finder optimization benchmark
tests/                      Greedy, DP and graph-cut tests
output/                     Generated results
```

## Adding an algorithm

1. Implement `run` in the relevant module under `src/algorithms/`.
2. Return `AlgorithmResult(image=..., algorithm=..., runtime_s=..., metadata=...)`.
3. If you add a new module, register its class in `src/algorithms/__init__.py`.

Input and output images are BGR `numpy.uint8` arrays, matching OpenCV.
