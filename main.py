"""Command line entry point for the seam carving project."""

import argparse
import sys

from src.algorithms import ALGORITHMS, get_algorithm
from src.config import DEFAULT_IMAGE, DEFAULT_OUTPUT
from src.display import show_image, show_pair
from src.image_io import load_image, save_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seam carving project runner")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="input image path (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default="dp",
        help="algorithm to run (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--seams",
        type=int,
        default=1,
        help="number of vertical seams to remove (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="only load and display the image, without running an algorithm",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the preview window",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        image = load_image(args.image)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    height, width = image.shape[:2]
    print(f"Loaded {args.image} ({width}x{height})")

    if args.preview:
        print("Preview only; no algorithm run.")
        if not args.no_show:
            show_image(image)
        return 0

    carver = get_algorithm(args.algorithm)
    print(f"Running '{carver.name}': {carver.description}")

    try:
        result = carver.run(image, num_seams=args.seams)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 1

    out_path = save_image(result.image, args.output)
    new_height, new_width = result.image.shape[:2]
    print(f"Saved {out_path} ({new_width}x{new_height}) in {result.runtime_s:.3f}s")

    if not args.no_show:
        show_pair(image, result.image, wait=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
