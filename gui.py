"""Tkinter GUI that exposes every seam carving setting in one window.

Launch with:  python main.py --gui   (or  python gui.py)

Keyboard shortcuts (same as main.py -i):
  A / Left   remove vertical seams (narrower)     Ctrl+Z  undo
  W / Up     remove horizontal seams (shorter)    D / Right  reset to original
  B          remove both directions               S / Ctrl+S save image
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from src.algorithms import ALGORITHMS, get_algorithm
from src.algorithms.graphcut import ENERGY_MODES, MAX_FLOW_METHODS
from src.config import DEFAULT_IMAGE, DEFAULT_OUTPUT
from src.image_io import load_image, save_image

DIRECTIONS = ("vertical", "horizontal", "both")
MAX_VIEW_W, MAX_VIEW_H = 1100, 650
MAX_UNDO = 20

# Slow algorithms: seconds for ONE seam on an image with `ref_pixels` pixels, and how
# the time grows with size (t ~ pixels ** exponent). Graph cut is fitted to the README
# timings (about 1 s on 143x97, 9 s on 286x194). Rough, machine-dependent estimates.
SLOW_ALGORITHMS = {
    "graphcut": {"ref_pixels": 143 * 97, "ref_seconds": 1.0, "exponent": 1.58},
}
SLOW_WARNING_SECONDS = 5.0  # ask before runs estimated to take longer than this


def estimate_seconds(algorithm: str, height: int, width: int, num_seams: int, direction: str) -> float:
    """Rough running time of a seam carving run; 0 for algorithms that are not slow."""
    model = SLOW_ALGORITHMS.get(algorithm)
    if model is None or num_seams <= 0:
        return 0.0

    def seam_time(h, w):
        return model["ref_seconds"] * (h * w / model["ref_pixels"]) ** model["exponent"]

    total = 0.0
    if direction in ("vertical", "both"):
        total += sum(seam_time(height, width - i) for i in range(num_seams))
        width -= num_seams
    if direction in ("horizontal", "both"):
        total += sum(seam_time(height - i, width) for i in range(num_seams))
    return total


def suggest_scale_pct(algorithm: str, height: int, width: int, num_seams: int, direction: str,
                      target_seconds: float = SLOW_WARNING_SECONDS) -> int | None:
    """Largest Scale % (step 5) of a height x width image at which removing the same
    num_seams fits target_seconds, or None if no scale does."""
    for pct in range(95, 0, -5):
        h, w = round(height * pct / 100), round(width * pct / 100)
        if num_seams >= {"vertical": w, "horizontal": h, "both": min(h, w)}[direction]:
            break  # image would be too small to remove that many seams
        if estimate_seconds(algorithm, h, w, num_seams, direction) <= target_seconds:
            return pct
    return None


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


class SeamCarvingGUI:
    def __init__(self, root: tk.Tk, image: np.ndarray | None = None, name: str | None = None):
        self.root = root
        self.root.title("Seam Carving")

        if image is None:
            image, name = load_image(DEFAULT_IMAGE), DEFAULT_IMAGE.name
        self.name = name or "image"
        self.loaded = image          # image exactly as opened (before scaling)
        self.original = image.copy()  # base image after "Scale", what Reset returns to
        self.current = image.copy()
        self.history: list[np.ndarray] = []
        self.last_result = None
        self.photo = None
        self.busy = False

        # ---- settings (one variable per CLI option) ----
        self.algorithm = tk.StringVar(value="dp")
        self.direction = tk.StringVar(value="vertical")
        self.num_seams = tk.IntVar(value=10)
        self.show_steps = tk.BooleanVar(value=False)
        self.delay_ms = tk.IntVar(value=0)
        self.multi_start = tk.BooleanVar(value=False)      # greedy only
        self.energy = tk.StringVar(value=ENERGY_MODES[0])  # graph cut only
        self.method = tk.StringVar(value=MAX_FLOW_METHODS[0])  # graph cut only
        self.scale_pct = tk.IntVar(value=100)
        self.compare = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
        self._on_algorithm_change()
        self._refresh()
        self._set_status(f"Loaded {self.name} ({self._size(self.current)})")

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        pad = {"padx": 4, "pady": 2}

        # Row 1: file / image
        row1 = ttk.Frame(self.root)
        row1.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 2))
        ttk.Button(row1, text="Open…", command=self.open_image).pack(side=tk.LEFT, **pad)
        ttk.Button(row1, text="Save…", command=self.save_image).pack(side=tk.LEFT, **pad)
        ttk.Button(row1, text="Save seams overlay…", command=self.save_overlay).pack(side=tk.LEFT, **pad)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(row1, text="Undo", command=self.undo).pack(side=tk.LEFT, **pad)
        ttk.Button(row1, text="Reset", command=self.reset_image).pack(side=tk.LEFT, **pad)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(row1, text="Scale %:").pack(side=tk.LEFT)
        ttk.Spinbox(row1, from_=5, to=100, increment=5, width=5, textvariable=self.scale_pct).pack(side=tk.LEFT)
        ttk.Button(row1, text="Apply scale", command=self.apply_scale).pack(side=tk.LEFT, **pad)
        ttk.Checkbutton(row1, text="Before / after", variable=self.compare, command=self._refresh).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Row 2: what to carve
        row2 = ttk.Frame(self.root)
        row2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(row2, text="Algorithm:").pack(side=tk.LEFT)
        algo = ttk.Combobox(row2, textvariable=self.algorithm, values=sorted(ALGORITHMS), state="readonly", width=9)
        algo.pack(side=tk.LEFT, **pad)
        algo.bind("<<ComboboxSelected>>", lambda e: self._on_algorithm_change())
        ttk.Label(row2, text="Direction:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Combobox(row2, textvariable=self.direction, values=DIRECTIONS, state="readonly", width=10).pack(
            side=tk.LEFT, **pad
        )
        ttk.Label(row2, text="Seams:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(row2, from_=0, to=5000, width=6, textvariable=self.num_seams).pack(side=tk.LEFT, **pad)
        self.remove_btn = ttk.Button(row2, text="Remove seams", command=self.remove_seams)
        self.remove_btn.pack(side=tk.LEFT, padx=(10, 4))
        ttk.Button(row2, text="Show all seams", command=self.show_all_seams).pack(side=tk.LEFT, **pad)

        # Row 3: how to carve (display + algorithm-specific options)
        row3 = ttk.Frame(self.root)
        row3.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        ttk.Checkbutton(row3, text="Step-by-step", variable=self.show_steps).pack(side=tk.LEFT, **pad)
        ttk.Label(row3, text="Delay ms:").pack(side=tk.LEFT)
        ttk.Spinbox(row3, from_=0, to=2000, increment=10, width=6, textvariable=self.delay_ms).pack(
            side=tk.LEFT, **pad
        )
        self.greedy_opts = ttk.Frame(row3)
        ttk.Checkbutton(self.greedy_opts, text="Multi-start greedy", variable=self.multi_start).pack(side=tk.LEFT)
        self.graphcut_opts = ttk.Frame(row3)
        ttk.Label(self.graphcut_opts, text="Energy:").pack(side=tk.LEFT)
        ttk.Combobox(
            self.graphcut_opts, textvariable=self.energy, values=ENERGY_MODES, state="readonly", width=9
        ).pack(side=tk.LEFT, **pad)
        ttk.Label(self.graphcut_opts, text="Max-flow:").pack(side=tk.LEFT)
        ttk.Combobox(
            self.graphcut_opts, textvariable=self.method, values=MAX_FLOW_METHODS, state="readonly", width=12
        ).pack(side=tk.LEFT, **pad)
        self.option_row = row3

        self.canvas = ttk.Label(self.root, relief=tk.SUNKEN, anchor="center")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8, pady=(0, 8))

    def _bind_keys(self):
        def shortcut(action):
            def handler(event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
                    return  # typing in a field (Spinbox is an Entry), not a shortcut
                action()
            return handler

        def carve(direction):
            def action():
                self.direction.set(direction)
                self.remove_seams()
            return shortcut(action)

        for seq in ("<Key-a>", "<Key-A>", "<Left>"):
            self.root.bind(seq, carve("vertical"))
        for seq in ("<Key-w>", "<Key-W>", "<Up>"):
            self.root.bind(seq, carve("horizontal"))
        for seq in ("<Key-b>", "<Key-B>"):
            self.root.bind(seq, carve("both"))
        for seq in ("<Key-d>", "<Key-D>", "<Right>"):
            self.root.bind(seq, shortcut(self.reset_image))
        for seq in ("<Key-s>", "<Key-S>"):
            self.root.bind(seq, shortcut(self.save_image))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_image())

    def _on_algorithm_change(self):
        self.greedy_opts.pack_forget()
        self.graphcut_opts.pack_forget()
        algo = self.algorithm.get()
        if algo == "greedy":
            self.greedy_opts.pack(side=tk.LEFT, padx=(12, 0))
        elif algo == "graphcut":
            self.graphcut_opts.pack(side=tk.LEFT, padx=(12, 0))
            self._set_status("Graph cut is slow: use 'Scale %' (e.g. 10-25) on large images first.")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _size(image: np.ndarray) -> str:
        return f"{image.shape[1]}x{image.shape[0]}"

    def _set_status(self, text: str):
        self.status.set(text)
        self.root.update_idletasks()

    def _to_photo(self, image: np.ndarray, max_w: int = MAX_VIEW_W, max_h: int = MAX_VIEW_H) -> tk.PhotoImage:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = min(1.0, max_w / w, max_h / h)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".ppm", image)
        if not ok:
            raise RuntimeError("failed to encode image for display")
        return tk.PhotoImage(data=buf.tobytes())

    def _render(self, image: np.ndarray):
        self.photo = self._to_photo(image)
        self.canvas.configure(image=self.photo)

    def _refresh(self):
        """Show the current image, or original | current side by side."""
        if self.compare.get():
            a, b = self.original, self.current
            h = max(a.shape[0], b.shape[0])
            pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            self._render(np.hstack([pad(a), pad(b)]))
        else:
            self._render(self.current)

    def _read_int(self, var: tk.IntVar, name: str) -> int | None:
        try:
            return int(var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid value", f"{name} must be a whole number.")
            return None

    def build_run_kwargs(self) -> dict:
        """Keyword arguments for carver.run() from the current settings."""
        kwargs = {"direction": self.direction.get(), "visualize": False, "show_all_seams": True}
        algo = self.algorithm.get()
        if algo == "greedy":
            kwargs["multi_start"] = bool(self.multi_start.get())
        elif algo == "graphcut":
            kwargs["energy"] = self.energy.get()
            kwargs["method"] = self.method.get()
        return kwargs

    def _step_callback(self, info):
        if self.show_steps.get():
            self._render(info["frame"])
        direction = f" {info['direction']}" if "direction" in info else ""
        self._set_status(f"Seam {info['step']}/{info['total']}{direction} | size {self._size(info['image'])}")
        self.root.update()  # keeps the window responsive during long (graph cut) runs
        delay = max(0, int(self.delay_ms.get() or 0)) if self.show_steps.get() else 0
        if delay:
            self.root.after(delay)

    # ------------------------------------------------------------------ actions

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.set_image(load_image(path), Path(path).name)
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def set_image(self, image: np.ndarray, name: str = "image"):
        self.name = name
        self.loaded = image
        self.original = image.copy()
        self.current = image.copy()
        self.history.clear()
        self.last_result = None
        self.scale_pct.set(100)
        self._refresh()
        self._set_status(f"Loaded {name} ({self._size(image)})")

    def apply_scale(self):
        if self.busy:
            return
        pct = self._read_int(self.scale_pct, "Scale %")
        if pct is None:
            return
        pct = min(100, max(1, pct))
        h, w = self.loaded.shape[:2]
        size = (max(2, round(w * pct / 100)), max(2, round(h * pct / 100)))
        scaled = self.loaded if pct == 100 else cv2.resize(self.loaded, size, interpolation=cv2.INTER_AREA)
        self.original = scaled.copy()
        self.current = scaled.copy()
        self.history.clear()
        self.last_result = None
        self._refresh()
        self._set_status(f"Scaled {self.name} to {pct}% -> {self._size(scaled)}")

    def save_image(self):
        self._save(self.current, "Save image", Path(DEFAULT_OUTPUT).name)

    def save_overlay(self):
        overlay = self.last_result.metadata.get("all_seams_image") if self.last_result else None
        if overlay is None:
            messagebox.showinfo("No seams yet", "Remove some seams first, then save the overlay.")
            return
        self._save(overlay, "Save seams overlay", f"{Path(DEFAULT_OUTPUT).stem}_all_seams.png")

    def _save(self, image: np.ndarray, title: str, initial: str):
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_image(image, path)
            self._set_status(f"Saved {path} ({self._size(image)})")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def reset_image(self):
        if self.busy:
            return
        self.current = self.original.copy()
        self.history.clear()
        self.last_result = None
        self._refresh()
        self._set_status(f"Reset to {self._size(self.current)}")

    def undo(self):
        if self.busy or not self.history:
            return
        self.current = self.history.pop()
        self.last_result = None
        self._refresh()
        self._set_status(f"Undo -> {self._size(self.current)} ({len(self.history)} more undo step(s))")

    def remove_seams(self):
        if self.busy:
            return
        n = self._read_int(self.num_seams, "Seams")
        if n is None:
            return
        direction = self.direction.get()
        h, w = self.current.shape[:2]
        limit = {"vertical": w, "horizontal": h, "both": min(h, w)}[direction]
        if not 0 < n < limit:
            side = {"vertical": "width", "horizontal": "height", "both": "width and height"}[direction]
            messagebox.showwarning("Cannot remove", f"Seams must be between 1 and {limit - 1} (image {side}).")
            return

        algo = self.algorithm.get()
        if not self.confirm_slow_run(algo, n, direction):
            self._set_status(f"[{algo}] cancelled")
            return
        kwargs = self.build_run_kwargs()
        self.busy = True
        self.remove_btn.state(["disabled"])
        try:
            result = get_algorithm(algo).run(self.current, num_seams=n, step_callback=self._step_callback, **kwargs)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._set_status(f"[{algo}] failed: {e}")
            return
        finally:
            self.busy = False
            self.remove_btn.state(["!disabled"])

        self.history.append(self.current)
        del self.history[:-MAX_UNDO]
        before = self._size(self.current)
        self.current = result.image
        self.last_result = result
        self._refresh()
        removed = f"{n} vertical + {n} horizontal" if direction == "both" else f"{n} {direction}"
        options = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k not in ("direction", "visualize", "show_all_seams"))
        self._set_status(
            f"[{algo}{f' {options}' if options else ''}] removed {removed} seam(s): "
            f"{before} -> {self._size(self.current)} in {result.runtime_s:.3f}s"
        )

    def confirm_slow_run(self, algo: str, n: int, direction: str) -> bool:
        """Warn before a run of a slow algorithm that will take a while. True = go ahead."""
        h, w = self.current.shape[:2]
        seconds = estimate_seconds(algo, h, w, n, direction)
        if seconds <= SLOW_WARNING_SECONDS:
            return True
        seams = f"{n} vertical + {n} horizontal" if direction == "both" else f"{n} {direction}"
        lines = [
            f"{algo} is slow. Removing {seams} seam(s) from a {w}x{h} image",
            f"will take roughly {format_duration(seconds)}, and the window will be busy until it finishes.",
            "",
        ]
        # 'Apply scale' always restarts from the image as opened, so suggest relative to that
        lh, lw = self.loaded.shape[:2]
        pct = suggest_scale_pct(algo, lh, lw, n, direction)
        if pct is not None:
            lines.append(
                f"Tip: set 'Scale %' to {pct} and click 'Apply scale' (restarts from the opened image "
                f"at {round(lw * pct / 100)}x{round(lh * pct / 100)}) to finish in a few seconds."
            )
        else:
            lines.append("Tip: shrink the image with 'Scale %' or remove fewer seams; dp gives the same seams much faster.")
        lines += ["", "Run anyway?"]
        return messagebox.askyesno("Slow algorithm", "\n".join(lines), icon=messagebox.WARNING, default=messagebox.NO)

    def show_all_seams(self):
        overlay = self.last_result.metadata.get("all_seams_image") if self.last_result else None
        if overlay is None:
            messagebox.showinfo("No seams yet", "Remove some seams first to see where they were.")
            return
        win = tk.Toplevel(self.root)
        win.title("All removed seams (last run, on its input image)")
        photo = self._to_photo(overlay)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)


def main():
    root = tk.Tk()
    try:
        SeamCarvingGUI(root)
    except FileNotFoundError as e:
        root.withdraw()
        messagebox.showerror("Seam Carving", str(e))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
