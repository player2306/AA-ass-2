"""Compare the original DP seam finder with the rolling-row implementation.

The old finder below is preserved only as an experimental baseline. The
production implementation lives in src/algorithms/dp.py. This measures the
finder on a pre-existing synthetic energy map, not end-to-end image resizing.

Run: python benchmark_q5_optimization.py --repeats 7
"""

import argparse
import csv
import random
import statistics
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.dp import _find_vertical_seam


def original_find_vertical_seam(energy):
    """Pre-optimization implementation with a full cost and int32 parent table."""
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)
    for i in range(1, h):
        left = np.roll(dp[i - 1], 1)
        left[0] = np.inf
        right = np.roll(dp[i - 1], -1)
        right[-1] = np.inf
        middle = dp[i - 1]
        choices = np.vstack((left, middle, right))
        idx = np.argmin(choices, axis=0) - 1
        backtrack[i] = np.arange(w) + idx
        dp[i] += choices[idx + 1, np.arange(w)]
    seam = np.empty(h, dtype=np.int32)
    seam[h - 1] = int(np.argmin(dp[h - 1]))
    for i in range(h - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]
    return seam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--width", type=int, default=700)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/q5_benchmark"))
    args = parser.parse_args()
    if min(args.height, args.width, args.repeats) < 1:
        parser.error("height, width and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    energy = np.random.default_rng(20261119).integers(0, 1024, (args.height, args.width)).astype(np.float32)
    versions = {"Original": original_find_vertical_seam, "Optimized": _find_vertical_seam}
    assert np.array_equal(original_find_vertical_seam(energy), _find_vertical_seam(energy))
    timings = {name: [] for name in versions}
    rng = random.Random(20261119)
    for _ in range(args.repeats):
        order = list(versions)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            versions[name](energy)
            timings[name].append((time.perf_counter() - start) * 1000)
    rows = []
    for name, finder in versions.items():
        # Trace allocations separately so tracing overhead does not affect timings.
        tracemalloc.start()
        finder(energy)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(implementation=name, height=args.height, width=args.width,
                         repeats=args.repeats, median_ms=statistics.median(timings[name]),
                         min_ms=min(timings[name]), max_ms=max(timings[name]),
                         traced_peak_mib=peak / 1024**2))
    with (args.output / "optimization_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
    for ax, key, title, unit in [(axes[0], "median_ms", "Median search time", "ms"),
                                  (axes[1], "traced_peak_mib", "Peak traced allocation", "MiB")]:
        values = [row[key] for row in rows]
        bars = ax.bar([row["implementation"] for row in rows], values,
                      color=["#9daab8", "#2166ac"], width=.55)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"DP seam search on a {args.height} × {args.width} energy map\n"
                 f"{args.repeats} timing runs; input map excluded from memory measurement")
    fig.savefig(args.output / "optimization_comparison.png", dpi=180)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
