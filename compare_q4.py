"""Question 4: compare Graph Cut (min-cut / max-flow) vs Dynamic Programming seam carving.

Produces in output/:
  q4_vertical_overlay.png     first vertical seam of each method
  q4_horizontal_overlay.png   first horizontal seam of each method
                              (green = DP, red = graph cut, yellow = same pixel)
  q4_{dp,graphcut,graphcut_forward}_n{N}.png         image after removing N seams
  q4_{dp,graphcut,graphcut_forward}_n{N}_seams.png   all N removed seams on the input
and prints tables of seam cost, running time and max-flow statistics.

The max-flow solver is plain Python, so by default the image is scaled down first
(--scale). DP runs on exactly the same scaled image, so the comparison is fair.

Run from the project root:
  python compare_q4.py [image] [--scale 0.1] [-n 5 10 20] [--sweep 0.05 0.1 0.15 0.2]
"""

import argparse
import time

import cv2
import numpy as np

from src.algorithms import get_algorithm
from src.algorithms.graphcut import (
    MAX_FLOW_METHODS,
    compute_cost_map,
    find_graphcut_horizontal_seam,
    find_graphcut_seam,
    seam_cost,
)
from src.config import DEFAULT_IMAGE, OUTPUT_DIR
from src.image_io import load_image, save_image
from src.utils import remove_vertical_seam

GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image.copy()
    h, w = image.shape[:2]
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def overlay_two_seams(image: np.ndarray, dp_seam: np.ndarray, gc_seam: np.ndarray, horizontal=False) -> np.ndarray:
    """DP seam in green, graph cut seam in red, shared pixels in yellow."""
    out = image.copy()
    idx = np.arange(len(dp_seam))
    if horizontal:
        out[dp_seam, idx] = GREEN
        out[gc_seam, idx] = RED
        out[dp_seam[dp_seam == gc_seam], idx[dp_seam == gc_seam]] = YELLOW
    else:
        out[idx, dp_seam] = GREEN
        out[idx, gc_seam] = RED
        out[idx[dp_seam == gc_seam], dp_seam[dp_seam == gc_seam]] = YELLOW
    return out


def replay_backward_cost(image: np.ndarray, seams: list[np.ndarray]) -> float:
    """Total Sobel (backward) energy of the removed seams, recomputing the energy after
    every removal. Gives one quality metric for every method, whatever energy it used."""
    total = 0.0
    img = image
    for seam in seams:
        total += seam_cost(compute_cost_map(img, "backward"), seam)
        img = remove_vertical_seam(img, seam)
    return total


def first_seam_comparison(image: np.ndarray, methods: list[str]) -> None:
    h, w = image.shape[:2]
    cost = compute_cost_map(image, "backward")

    print("== First vertical seam (same Sobel energy map) ==")
    start = time.perf_counter()
    dp_seam = get_algorithm("dp").run(image, num_seams=1).metadata["seams"][0]
    dp_time = time.perf_counter() - start
    dp_cost = seam_cost(cost, dp_seam)
    print(f"{'method':<22} {'seam cost':>10} {'max-flow':>10} {'time (s)':>9} {'BFS runs':>9} "
          f"{'aug. paths':>10} {'px != DP':>9}")
    print(f"{'dp':<22} {dp_cost:>10.0f} {'-':>10} {dp_time:>9.3f} {'-':>9} {'-':>10} {'-':>9}")

    gc_seam = None
    for method in methods:
        start = time.perf_counter()
        seam, info = find_graphcut_seam(cost, method=method, return_info=True)
        elapsed = time.perf_counter() - start
        gc_seam = seam if gc_seam is None else gc_seam
        diff = int((seam != dp_seam).sum())
        print(f"{'graphcut/' + method:<22} {seam_cost(cost, seam):>10.0f} {info['max_flow']:>10.0f} "
              f"{elapsed:>9.3f} {info['bfs_runs']:>9} {info['augmenting_paths']:>10} {diff:>9}")
    print(f"graph: {h * w + 2} nodes, {info['edges']} edges "
          f"(pixels {w}x{h})")
    save_image(overlay_two_seams(image, dp_seam, gc_seam), OUTPUT_DIR / "q4_vertical_overlay.png")

    print("\n== First horizontal seam (DP on the transposed image) ==")
    transposed = np.ascontiguousarray(np.transpose(image, (1, 0, 2)))
    dp_h = get_algorithm("dp").run(transposed, num_seams=1).metadata["seams"][0]
    gc_h, info = find_graphcut_horizontal_seam(cost, method=methods[0], return_info=True)
    cols = np.arange(w)
    print(f"dp seam cost {cost[dp_h, cols].sum():.0f} | graphcut seam cost {cost[gc_h, cols].sum():.0f} "
          f"| max-flow {info['max_flow']:.0f} | pixels that differ {int((dp_h != gc_h).sum())}")
    save_image(overlay_two_seams(image, dp_h, gc_h, horizontal=True), OUTPUT_DIR / "q4_horizontal_overlay.png")


def multi_seam_comparison(image: np.ndarray, counts: list[int], method: str) -> None:
    print("\n== Removing N vertical seams ==")
    print("total cost = Sobel energy of the removed seams (recomputed after each removal); lower is better.")
    print("graphcut_forward minimises a different cost (forward energy), so judge it on the saved images.")
    print(f"{'algorithm':<18} {'N':>4} {'runtime (s)':>12} {'s / seam':>9} {'total cost':>11} {'seams == DP':>12}")
    runs = [
        ("dp", "dp", {}),
        ("graphcut", "graphcut", {"method": method}),
        ("graphcut_forward", "graphcut", {"method": method, "energy": "forward"}),
    ]
    for n in counts:
        if n >= image.shape[1]:
            print(f"skip N={n}: image is only {image.shape[1]} px wide")
            continue
        dp_seams = None
        for label, algo, kwargs in runs:
            result = get_algorithm(algo).run(image, num_seams=n, show_all_seams=True, **kwargs)
            seams = result.metadata["seams_original"]
            if label == "dp":
                dp_seams = seams
                same = "-"
            else:
                same = f"{sum(np.array_equal(a, b) for a, b in zip(seams, dp_seams))}/{n}"
            total = replay_backward_cost(image, result.metadata["seams"])
            print(f"{label:<18} {n:>4} {result.runtime_s:>12.3f} {result.runtime_s / max(n, 1):>9.3f} "
                  f"{total:>11.0f} {same:>12}")
            save_image(result.image, OUTPUT_DIR / f"q4_{label}_n{n}.png")
            save_image(result.metadata["all_seams_image"], OUTPUT_DIR / f"q4_{label}_n{n}_seams.png")


def size_sweep(image: np.ndarray, scales: list[float], method: str) -> None:
    print("\n== Running time vs image size (one vertical seam) ==")
    print(f"{'size':>10} {'pixels':>8} {'DP (ms)':>9} {'graph cut (s)':>14} {'ratio':>8} {'BFS runs':>9} {'aug. paths':>10}")
    for scale in scales:
        img = scale_image(image, scale)
        h, w = img.shape[:2]
        dp = get_algorithm("dp").run(img, num_seams=1)
        gc = get_algorithm("graphcut").run(img, num_seams=1, method=method)
        stats = gc.metadata["solver_stats"][0]
        print(f"{f'{w}x{h}':>10} {h * w:>8} {dp.runtime_s * 1000:>9.2f} {gc.runtime_s:>14.3f} "
              f"{gc.runtime_s / dp.runtime_s:>7.0f}x {stats['bfs_runs']:>9} {stats['augmenting_paths']:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4: Graph cut vs DP comparison")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("--scale", type=float, default=0.1, help="resize factor before carving (default: %(default)s)")
    parser.add_argument("-n", "--seams", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--method", choices=MAX_FLOW_METHODS, default="dinic",
                        help="max-flow algorithm for the N-seam runs (default: %(default)s)")
    parser.add_argument("--first-seam-methods", nargs="+", choices=MAX_FLOW_METHODS, default=list(MAX_FLOW_METHODS),
                        help="max-flow algorithms timed on the first seam")
    parser.add_argument("--sweep", type=float, nargs="*", default=[],
                        help="extra scales for a running-time vs size table, e.g. --sweep 0.05 0.1 0.15 0.2")
    args = parser.parse_args()

    original = load_image(args.image)
    image = scale_image(original, args.scale)
    print(f"Image: {args.image} {original.shape[1]}x{original.shape[0]} "
          f"-> scale {args.scale} -> {image.shape[1]}x{image.shape[0]}\n")

    first_seam_comparison(image, args.first_seam_methods)
    multi_seam_comparison(image, args.seams, args.method)
    if args.sweep:
        size_sweep(original, args.sweep, args.method)

    print(f"\nImages saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
