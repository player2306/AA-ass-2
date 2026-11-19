"""Command line entry point for the seam carving project."""

import argparse
import sys
from pathlib import Path

import cv2

from src.algorithms import ALGORITHMS, carve_to_size, get_algorithm
from src.config import DEFAULT_IMAGE, DEFAULT_OUTPUT
from src.display import fit_to_window, show_image, show_pair
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
        help="number of seams to remove (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=["vertical", "horizontal", "both"],
        default="vertical",
        help="vertical seams make the image narrower, horizontal seams make it shorter, "
        "both removes n vertical then n horizontal (default: %(default)s)",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="carve until the image is this wide (replaces -n/-d; can be combined with --height)",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="carve until the image is this tall (replaces -n/-d; can be combined with --width)",
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show step-by-step seam removal visualization",
    )
    parser.add_argument(
        "--step-delay",
        type=int,
        default=1,
        help="milliseconds to wait between visualization steps (0 waits for key)",
    )
    parser.add_argument(
        "--show-all-seams",
        action="store_true",
        help="display/save all removed seams overlaid on the original image",
    )
    parser.add_argument(
        "--all-seams-output",
        help="optional path to save the all-seams visualization image",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="interactively edit image size with keyboard controls",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="number of seams removed per interactive step (default: %(default)s)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch simple Tkinter GUI for seam carving",
    )
    return parser.parse_args(argv)


def _save_all_seams(result, output_path: str | None, fallback_output: str) -> None:
    overlay = result.metadata.get("all_seams_image")
    if overlay is None:
        return
    if output_path:
        out = output_path
    else:
        p = Path(fallback_output)
        out = str(p.with_name(f"{p.stem}_all_seams{p.suffix}"))
    saved = save_image(overlay, out)
    print(f"Saved all-seams overlay {saved}")


def run_interactive(image, args) -> int:
    carver = get_algorithm(args.algorithm)
    current = image.copy()
    original = image.copy()
    seams_done = 0
    step = max(1, args.step_size)
    win = "Interactive Seam Carving"

    help_lines = (
        "[A/Left] narrower  [W/Up] shorter  [D/Right] reset  [S] save  [+/-] step  [Q/Esc] quit"
    )
    print("Interactive controls:")
    print("  A or Left Arrow : remove vertical seams (narrower)")
    print("  W or Up Arrow   : remove horizontal seams (shorter)")
    print("  D or Right Arrow: reset to original")
    print("  + / -           : change step size")
    print("  S               : save current image")
    print("  Q or Esc        : quit")

    while True:
        h, w = current.shape[:2]
        frame = current.copy()
        text = f"{w}x{h} | removed: {seams_done} | step: {step}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, help_lines, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, fit_to_window(frame))
        key = cv2.waitKeyEx(0)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("a"), ord("A"), 2424832, ord("w"), ord("W"), 2490368):  # left / up
            direction = "horizontal" if key in (ord("w"), ord("W"), 2490368) else "vertical"
            size = h if direction == "horizontal" else w
            if size <= step:
                print(f"Cannot remove more seams; at minimum {'height' if direction == 'horizontal' else 'width'}.")
                continue
            n = min(step, size - 1)
            result = carver.run(
                current,
                num_seams=n,
                direction=direction,
                visualize=args.visualize,
                step_delay=args.step_delay,
                show_all_seams=args.show_all_seams or bool(args.all_seams_output),
            )
            current = result.image
            seams_done += n
            if args.show_all_seams or args.all_seams_output:
                tag = Path(args.output)
                overlay_path = str(tag.with_name(f"{tag.stem}_interactive_{seams_done}_seams{tag.suffix}"))
                _save_all_seams(result, args.all_seams_output and overlay_path, overlay_path)
            print(f"Removed {n} seam(s) -> {current.shape[1]}x{current.shape[0]} ({result.runtime_s:.3f}s)")
        elif key in (ord("d"), ord("D"), 2555904):  # right
            current = original.copy()
            seams_done = 0
            print(f"Reset to original {original.shape[1]}x{original.shape[0]}")
        elif key in (ord("s"), ord("S")):
            out_path = save_image(current, args.output)
            print(f"Saved {out_path} ({current.shape[1]}x{current.shape[0]})")
        elif key in (ord("+"), ord("=")):
            step += 1
            print(f"Step size: {step}")
        elif key in (ord("-"), ord("_")) and step > 1:
            step -= 1
            print(f"Step size: {step}")

    cv2.destroyWindow(win)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.gui:
        from gui import main as gui_main

        gui_main()
        return 0

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

    if args.interactive:
        return run_interactive(image, args)

    carver = get_algorithm(args.algorithm)
    print(f"Running '{carver.name}': {carver.description}")

    options = dict(
        visualize=args.visualize,
        step_delay=args.step_delay,
        show_all_seams=args.show_all_seams or bool(args.all_seams_output),
    )
    try:
        if args.width is not None or args.height is not None:
            target_w = width if args.width is None else args.width
            target_h = height if args.height is None else args.height
            print(f"Carving to {target_w}x{target_h}")
            result = carve_to_size(carver, image, target_w, target_h, **options)
        else:
            result = carver.run(image, num_seams=args.seams, direction=args.direction, **options)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # e.g. too many seams for the image size
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        out_path = save_image(result.image, args.output)
    except OSError as exc:  # e.g. unsupported output extension
        print(f"error: {exc}", file=sys.stderr)
        return 1
    new_height, new_width = result.image.shape[:2]
    print(f"Saved {out_path} ({new_width}x{new_height}) in {result.runtime_s:.3f}s")

    if args.show_all_seams or args.all_seams_output:
        _save_all_seams(result, args.all_seams_output, args.output)

    if not args.no_show and not args.visualize:
        if args.show_all_seams and result.metadata.get("all_seams_image") is not None:
            cv2.imshow("All Seams (on original)", fit_to_window(result.metadata["all_seams_image"]))
            show_pair(image, result.image, wait=True)
            cv2.destroyAllWindows()
        else:
            show_pair(image, result.image, wait=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
