"""Question 5: reproducible end-to-end seam-carving timing comparison.

Run from the project root after installing requirements and matplotlib:
    python benchmark_q5.py --repeats 3

The graphs use a log y-axis because graph cut can be much slower. Each timed
call includes energy calculation, seam search, deletion, and result bookkeeping;
visualization is disabled. Graph cut uses its default Dinic solver and backward
Sobel energy, and greedy uses the default single-start implementation.
"""

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms import get_algorithm
from src.algorithms.graphcut import compute_cost_map, seam_cost
from src.config import DEFAULT_IMAGE
from src.image_io import load_image


NAMES = ("dp", "greedy", "graphcut")
LABELS = {"dp": "Dynamic programming", "greedy": "Greedy", "graphcut": "Graph cut (Dinic)"}
COLORS = {"dp": "#2166ac", "greedy": "#e08214", "graphcut": "#b2182b"}


def resized(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (max(2, round(w * scale)), max(2, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_case(image, count, repeats, rng, experiment):
    """All measurements use the same image and count, with shuffled order."""
    h, w = image.shape[:2]
    cost = compute_cost_map(image, "backward")
    rows = []
    for repeat in range(repeats):
        order = list(NAMES)
        rng.shuffle(order)
        for name in order:
            carver = get_algorithm(name)
            start = time.perf_counter()
            result = carver.run(image, num_seams=count, direction="vertical",
                                visualize=False, show_all_seams=False, keep_steps=False)
            elapsed_ms = (time.perf_counter() - start) * 1000
            seam = result.metadata["seams"][0]
            if len(seam) != h or np.any(seam < 0) or np.any(seam >= w) or np.any(np.abs(np.diff(seam)) > 1):
                raise ValueError(f"Invalid {name} seam for {w}x{h}")
            rows.append(dict(experiment=experiment, width=w, height=h, pixels=w*h,
                             seams=count, repeat=repeat+1, algorithm=name,
                             elapsed_ms=elapsed_ms, first_seam_cost=seam_cost(cost, seam)))
        by_name = {row["algorithm"]: row["first_seam_cost"] for row in rows[-len(NAMES):]}
        if not np.isclose(by_name["dp"], by_name["graphcut"], atol=1e-4):
            raise AssertionError(f"DP and graph-cut costs differ: {by_name}")
        if by_name["greedy"] + 1e-4 < by_name["dp"]:
            raise AssertionError(f"Greedy cost unexpectedly lower than DP: {by_name}")
    print(f"{experiment}: {w}x{h}, {count} seam(s), {repeats} repeats", flush=True)
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["experiment"], r["width"], r["height"], r["pixels"], r["seams"], r["algorithm"])
        groups.setdefault(key, []).append(r)
    result = []
    for key, group in groups.items():
        experiment, w, h, pixels, count, name = key
        times = [r["elapsed_ms"] for r in group]
        result.append(dict(experiment=experiment, width=w, height=h, pixels=pixels,
                           seams=count, algorithm=name, repeats=len(times),
                           median_ms=statistics.median(times), min_ms=min(times),
                           max_ms=max(times), first_seam_cost=group[0]["first_seam_cost"]))
    return result


def plot(summary, experiment, xkey, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for name in NAMES:
        rows = sorted((r for r in summary if r["experiment"] == experiment and r["algorithm"] == name),
                      key=lambda r: r[xkey])
        ax.plot([r[xkey] for r in rows], [r["median_ms"] for r in rows], "o-",
                color=COLORS[name], label=LABELS[name], linewidth=2, markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median end-to-end time (ms, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.03, 0.05, 0.075, 0.1])
    parser.add_argument("--count-scale", type=float, default=0.05)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path, default=Path("output/q5_benchmark"))
    args = parser.parse_args()
    if args.repeats < 1 or any(s <= 0 for s in args.scales + [args.count_scale]) or any(n < 1 for n in args.counts):
        parser.error("repeats, scales and seam counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    image = load_image(args.image)
    rng = random.Random(20261119)
    # One unmeasured call per algorithm reduces first-call effects.
    warm = resized(image, 0.02)
    for name in NAMES:
        get_algorithm(name).run(warm, num_seams=1, visualize=False, show_all_seams=False)
    rows = []
    for scale in args.scales:
        rows.extend(run_case(resized(image, scale), 1, args.repeats, rng, "image_size"))
    count_image = resized(image, args.count_scale)
    for count in args.counts:
        rows.extend(run_case(count_image, count, args.repeats, rng, "seam_count"))
    summary = summarize(rows)
    write_csv(args.output / "raw_timings.csv", rows)
    write_csv(args.output / "median_timings.csv", summary)
    plot(summary, "image_size", "pixels", "Image size (pixels)",
         "Time to remove one vertical seam", args.output / "time_vs_image_size.png")
    h, w = count_image.shape[:2]
    plot(summary, "seam_count", "seams", "Vertical seams removed",
         f"Time versus seams removed ({w} × {h} image)", args.output / "time_vs_seam_count.png")
    info = dict(image=str(Path(args.image).resolve()), original_shape=list(image.shape),
                python=sys.version, platform=platform.platform(), processor=platform.processor(),
                numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__,
                opencv_threads=cv2.getNumThreads(), repeats=args.repeats, scales=args.scales,
                count_scale=args.count_scale, counts=args.counts, direction="vertical",
                graphcut_solver="dinic", graphcut_energy="backward Sobel", greedy="single-start",
                time_scope="wall clock around entire run(); input loading and resizing excluded",
                order="shuffled with seed 20261119; one warmup per algorithm")
    (args.output / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
