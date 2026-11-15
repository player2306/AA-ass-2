"""Question 3: compare Greedy vs Dynamic Programming seam carving.

Produces in output/:
  q3_seam_overlay.png   first seam of each method on the original image
                        (red = greedy, green = DP)
  q3_greedy_n{N}.png    image after removing N seams with greedy
  q3_dp_n{N}.png        image after removing N seams with DP
and prints a table of seam energy and runtime.

Run from the project root:  python compare_q3.py [image] [-n 10 50 100]
"""

import argparse

from src.algorithms import get_algorithm
from src.algorithms.greedy import find_greedy_seam
from src.config import DEFAULT_IMAGE, OUTPUT_DIR
from src.image_io import load_image, save_image
from src.seam_utils import compute_energy, draw_seam, seam_energy

try:  # Member 1 is expected to expose this in dp.py
    from src.algorithms.dp import find_dp_seam
except ImportError:
    find_dp_seam = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Greedy vs DP comparison")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    parser.add_argument("-n", "--seams", type=int, nargs="+", default=[10, 50, 100])
    args = parser.parse_args()

    image = load_image(args.image)
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})\n")

    # 1. Same energy map for both -> compare the first seam each method picks
    energy = compute_energy(image)
    greedy_seam = find_greedy_seam(energy)
    overlay = draw_seam(image, greedy_seam, color=(0, 0, 255))  # red
    print(f"First seam energy  greedy: {seam_energy(energy, greedy_seam):10.1f}")
    if find_dp_seam is not None:
        dp_seam = find_dp_seam(energy)
        overlay = draw_seam(overlay, dp_seam, color=(0, 255, 0))  # green
        print(f"First seam energy  DP    : {seam_energy(energy, dp_seam):10.1f}")
    else:
        print("DP seam not available yet (add find_dp_seam(energy) to dp.py)")
    save_image(overlay, OUTPUT_DIR / "q3_seam_overlay.png")

    # 2. Remove N seams with each algorithm -> runtime + total energy removed
    print(f"\n{'algorithm':<8} {'seams':>6} {'runtime (s)':>12} {'total seam energy':>18}")
    for n in args.seams:
        for name in ("greedy", "dp"):
            try:
                result = get_algorithm(name).run(image, num_seams=n)
            except NotImplementedError:
                print(f"{name:<8} {n:>6} {'not implemented':>12}")
                continue
            total = sum(result.metadata.get("seam_energies", [])) or float("nan")
            print(f"{name:<8} {n:>6} {result.runtime_s:>12.3f} {total:>18.1f}")
            save_image(result.image, OUTPUT_DIR / f"q3_{name}_n{n}.png")

    print(f"\nImages saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
