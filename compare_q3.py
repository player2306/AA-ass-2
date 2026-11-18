"""Question 3: compare Greedy vs Dynamic Programming seam carving.

Run from the project root:
    python compare_q3.py [image] [-n 10 50 100] [-r 5]

What it measures:
  1. First seam on the SAME energy map: total energy of the Greedy and DP seam.
  2. Per-seam cost of each step (energy, seam finding, removal), median of
     r repeats. Seam finding is where Greedy and DP really differ.
  3. Removing N seams with each algorithm's run(): median runtime of r repeats
     (the order alternates each repeat, so neither algorithm always goes first)
     and the total energy of the removed seams.

Files written to output/:
  q3_seam_overlay.png        first seams: red = greedy, green = DP, yellow = both
  q3_seam_overlay_thick.png  same, drawn thicker so it is visible in the report
  q3_{alg}_n{N}.png          image after removing N seams
  q3_{alg}_all_seams_n{N}.png  all N removed seams drawn on the original image
  q3_first_seam.csv          table 1
  q3_step_timing.csv         table 2
  q3_runs.csv                table 3
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from src.algorithms import ALGORITHMS, get_algorithm
from src.algorithms.greedy import find_greedy_seam, find_multistart_greedy_seam
from src.config import DEFAULT_IMAGE, OUTPUT_DIR
from src.image_io import load_image, save_image
from src.seam_utils import seam_energy
from src.utils import remove_vertical_seam, sobel_edge_map

# DP's seam finder (energy -> seam). Use the public name if Member 1 adds one,
# otherwise their existing internal function, which has the same input/output.
try:
    from src.algorithms.dp import find_dp_seam
except ImportError:
    try:
        from src.algorithms.dp import _find_vertical_seam as find_dp_seam
    except ImportError:
        find_dp_seam = None

RED, GREEN, YELLOW = (0, 0, 255), (0, 255, 0), (0, 255, 255)  # BGR colours


# ---------------------------------------------------------------- helpers

def median_time(func, repeats: int) -> float:
    """Run func() `repeats` times and return the median time in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def seam_energies_of(result, image: np.ndarray) -> list[float] | None:
    """Energy of every seam an algorithm removed, or None if it can't be worked out.

    Greedy stores these directly in metadata["seam_energies"]. DP only stores the
    seams themselves (metadata["seams"]), so for DP we replay the run: for each
    seam, compute the energy of the current image, add up the energy along the
    seam, then remove it. This happens outside the timed run.
    """
    if "seam_energies" in result.metadata:
        return list(result.metadata["seam_energies"])
    seams = result.metadata.get("seams")
    if seams is None:
        return None

    energies = []
    img = image.copy()
    for seam in seams:
        energies.append(seam_energy(sobel_edge_map(img), seam))
        img = remove_vertical_seam(img, seam)
    return energies


def seam_similarity(seam_a: np.ndarray, seam_b: np.ndarray) -> dict:
    """How alike two vertical seams are, row by row.

    same_rows_pct      % of rows where both seams remove the same pixel
    mean_col_distance  average horizontal gap between the seams (pixels)
    max_col_distance   largest horizontal gap between the seams (pixels)
    """
    gap = np.abs(np.asarray(seam_a) - np.asarray(seam_b))
    return {
        "same_rows_pct": float((gap == 0).mean() * 100),
        "mean_col_distance": float(gap.mean()),
        "max_col_distance": int(gap.max()),
    }


def seam_overlay(image, greedy_seam, dp_seam=None, thickness: int = 1) -> np.ndarray:
    """Draw the greedy seam in red and the DP seam in green; pixels on both are yellow.

    thickness > 1 widens the lines sideways. That is only for viewing: the
    real seams are always exactly one pixel per row.
    """
    height, width = image.shape[:2]
    rows = np.arange(height)

    def mask_of(seam):
        mask = np.zeros((height, width), dtype=np.uint8)
        if seam is not None:
            mask[rows, seam] = 1
            if thickness > 1:
                mask = cv2.dilate(mask, np.ones((1, thickness), np.uint8))
        return mask.astype(bool)

    greedy, dp = mask_of(greedy_seam), mask_of(dp_seam)
    out = image.copy()
    out[greedy & ~dp] = RED
    out[dp & ~greedy] = GREEN
    out[greedy & dp] = YELLOW
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as a CSV file (column names from the first row)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def size_experiment(image: np.ndarray, scales: list[float], repeats: int) -> list[dict]:
    """Time one seam's steps on resized copies of the image.

    If the Big-O claims hold, then as the image grows:
      - greedy seam time / H stays roughly constant  (Greedy is O(H + W), mostly H)
      - DP seam time / (H*W) stays roughly constant  (DP is O(H*W))
      - energy and removal time / (H*W) stay roughly constant
    """
    print("\n4. One seam at different image sizes (ms, median)")
    print(f"   {'scale':>5} {'W x H':>11} {'energy':>8} {'greedy':>8} {'multi':>8} {'dp':>8}"
          f" {'remove':>8} {'greedy us/row':>14} {'dp ns/pixel':>12}")
    rows = []
    for scale in scales:
        img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = img.shape[:2]
        energy = sobel_edge_map(img)
        seam = find_greedy_seam(energy)

        greedy_ms = median_time(lambda: find_greedy_seam(energy), repeats) * 1e3
        dp_ms = median_time(lambda: find_dp_seam(energy), repeats) * 1e3 if find_dp_seam else None
        row = {
            "scale": scale,
            "width": width,
            "height": height,
            "pixels": width * height,
            "energy_ms": median_time(lambda: sobel_edge_map(img), repeats) * 1e3,
            "greedy_seam_ms": greedy_ms,
            "greedy_multi_seam_ms": median_time(
                lambda: find_multistart_greedy_seam(energy), repeats) * 1e3,
            "dp_seam_ms": dp_ms,
            "removal_ms": median_time(lambda: remove_vertical_seam(img, seam), repeats) * 1e3,
            # time divided by the size Big-O says it should grow with
            "greedy_us_per_row": greedy_ms * 1e3 / height,
            "dp_ns_per_pixel": dp_ms * 1e6 / (width * height) if dp_ms is not None else None,
        }
        rows.append(row)
        dp_text = f"{dp_ms:8.2f}" if dp_ms is not None else f"{'n/a':>8}"
        per_px = f"{row['dp_ns_per_pixel']:12.1f}" if dp_ms is not None else f"{'n/a':>12}"
        print(f"   {scale:5.2f} {f'{width}x{height}':>11} {row['energy_ms']:8.2f} {greedy_ms:8.2f}"
              f" {row['greedy_multi_seam_ms']:8.2f} {dp_text} {row['removal_ms']:8.2f} {row['greedy_us_per_row']:14.2f} {per_px}")
    return rows


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Greedy vs DP comparison")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("-n", "--seams", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("-r", "--repeats", type=int, default=5,
                        help="repeats per measurement; the median is reported")
    parser.add_argument("-s", "--scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                        help="image scales for the size experiment (section 4)")
    args = parser.parse_args()
    repeats = max(1, args.repeats)

    image = load_image(args.image)
    height, width = image.shape[:2]
    print(f"Image: {args.image} ({width}x{height}), {repeats} repeats per measurement\n")

    # What to run in section 3: label -> (registered algorithm name, extra run() options).
    # DP may be missing or not registered, so only keep what exists.
    variants = {
        "greedy": ("greedy", {}),
        "greedy_multi": ("greedy", {"multi_start": True}),
        "dp": ("dp", {}),
    }
    variants = {label: v for label, v in variants.items() if v[0] in ALGORITHMS}
    algorithms = list(variants)
    if "dp" not in algorithms:
        print("Note: 'dp' is not registered, comparing greedy only.\n")
    if find_dp_seam is None:
        print("Note: DP seam finder not found in dp.py, first-seam comparison is greedy only.\n")

    # ---- 1. First seam on the same energy map
    energy = sobel_edge_map(image)
    greedy_seam = find_greedy_seam(energy)
    dp_seam = find_dp_seam(energy) if find_dp_seam else None

    multi_seam = find_multistart_greedy_seam(energy)
    first = {
        "greedy_energy": seam_energy(energy, greedy_seam),
        "greedy_multi_energy": seam_energy(energy, multi_seam),
    }
    if dp_seam is not None:
        first["dp_energy"] = seam_energy(energy, dp_seam)
        first.update(seam_similarity(greedy_seam, dp_seam))
        first["energy_ratio_greedy_over_dp"] = (
            first["greedy_energy"] / first["dp_energy"] if first["dp_energy"] else None)
    first_rows = [first]

    print("1. First seam (same energy map)")
    print(f"   greedy total energy   {first['greedy_energy']:12.1f}")
    print(f"   multi-start greedy    {first['greedy_multi_energy']:12.1f}")
    if dp_seam is not None:
        ratio = first["energy_ratio_greedy_over_dp"]
        print(f"   DP total energy       {first['dp_energy']:12.1f}")
        print(f"   greedy / DP energy    {ratio:12.2f}x" if ratio is not None else
              "   greedy / DP energy             n/a (DP seam energy is 0)")
        print(f"   rows where seams match{first['same_rows_pct']:11.1f} %")
        print(f"   mean column distance  {first['mean_col_distance']:12.1f} px"
              f"   (max {first['max_col_distance']} px)")
    save_image(seam_overlay(image, greedy_seam, dp_seam), OUTPUT_DIR / "q3_seam_overlay.png")
    save_image(seam_overlay(image, greedy_seam, dp_seam, thickness=5),
               OUTPUT_DIR / "q3_seam_overlay_thick.png")

    # ---- 2. Cost of each step for one seam (median of repeats)
    # Energy and removal use the same shared functions for both algorithms.
    energy_ms = median_time(lambda: sobel_edge_map(image), repeats) * 1e3
    removal_ms = median_time(lambda: remove_vertical_seam(image, greedy_seam), repeats) * 1e3
    finders = {"greedy": find_greedy_seam, "greedy_multi": find_multistart_greedy_seam}
    if find_dp_seam is not None:
        finders["dp"] = find_dp_seam

    step_rows = []
    for name, finder in finders.items():
        step_rows.append({
            "algorithm": name,
            "energy_ms": energy_ms,
            "seam_find_ms": median_time(lambda: finder(energy), repeats) * 1e3,
            "removal_ms": removal_ms,
        })

    print("\n2. Time per step for one seam (ms, median)")
    print(f"   {'algorithm':<12} {'energy':>8} {'find seam':>10} {'remove':>8}")
    for row in step_rows:
        print(f"   {row['algorithm']:<12} {row['energy_ms']:8.2f} "
              f"{row['seam_find_ms']:10.2f} {row['removal_ms']:8.2f}")

    # ---- 3. Remove N seams with run(): median runtime + total seam energy
    print("\n3. Removing N seams with run() (runtime = median)")
    print(f"   {'algorithm':<12} {'seams':>6} {'runtime (s)':>12} {'total seam energy':>18}")
    run_rows = []
    for n in args.seams:
        if not 0 <= n < width:
            print(f"   skipping n={n}: must be between 0 and {width - 1} for this image")
            continue

        times = {name: [] for name in algorithms}
        last_result = {}
        for rep in range(repeats):
            # alternate the order so neither algorithm always runs first
            order = algorithms if rep % 2 == 0 else algorithms[::-1]
            for name in order:
                if name not in times:  # dropped earlier (not implemented)
                    continue
                algorithm_name, options = variants[name]
                try:
                    result = get_algorithm(algorithm_name).run(
                        image, num_seams=n, show_all_seams=True, **options)
                except NotImplementedError:
                    print(f"   {name}: not implemented, skipped")
                    del times[name]
                    continue
                times[name].append(result.runtime_s)
                last_result[name] = result

        for name, runtimes in times.items():
            result = last_result[name]
            energies = seam_energies_of(result, image)
            total = sum(energies) if energies is not None else None
            timing = result.metadata.get("timing", {})
            run_rows.append({
                "algorithm": name,
                "seams": n,
                "runtime_median_s": statistics.median(runtimes),
                "runtime_min_s": min(runtimes),
                "runtime_max_s": max(runtimes),
                "total_seam_energy": total,
                "mean_seam_energy": total / n if total is not None and n else None,
                "seam_energy_status": "ok" if energies is not None else "missing",
                # breakdown of the last run (only greedy reports it)
                "energy_s": timing.get("energy_s"),
                "seam_s": timing.get("seam_s"),
                "removal_s": timing.get("removal_s"),
            })
            total_text = f"{total:18.1f}" if total is not None else f"{'n/a (missing)':>18}"
            print(f"   {name:<12} {n:>6} {statistics.median(runtimes):12.3f} {total_text}")

            save_image(result.image, OUTPUT_DIR / f"q3_{name}_n{n}.png")
            if result.metadata.get("all_seams_image") is not None:
                save_image(result.metadata["all_seams_image"],
                           OUTPUT_DIR / f"q3_{name}_all_seams_n{n}.png")

    # ---- 4. Same per-seam steps at several image sizes (backs up the Big-O)
    size_rows = size_experiment(image, args.scales, repeats)

    # ---- CSV files for report tables / plots
    print()
    write_csv(OUTPUT_DIR / "q3_first_seam.csv", first_rows)
    write_csv(OUTPUT_DIR / "q3_step_timing.csv", step_rows)
    write_csv(OUTPUT_DIR / "q3_runs.csv", run_rows)
    write_csv(OUTPUT_DIR / "q3_sizes.csv", size_rows)
    print(f"Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
