import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from src.algorithms import ALGORITHMS, get_algorithm
from src.config import DEFAULT_IMAGE, DEFAULT_OUTPUT
from src.image_io import load_image, save_image


class SeamCarvingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Seam Carving GUI")

        self.original = load_image(DEFAULT_IMAGE)
        self.current = self.original.copy()
        self.last_result = None
        self.photo = None

        self.status = tk.StringVar(value="")
        self.step_size = tk.IntVar(value=5)
        self.delay_ms = tk.IntVar(value=50)
        self.show_steps = tk.BooleanVar(value=True)
        self.algorithm = tk.StringVar(value="dp")

        self._build_ui()
        self._render(self.current)
        self._set_status(f"Loaded {DEFAULT_IMAGE.name} ({self.current.shape[1]}x{self.current.shape[0]})")

    def _build_ui(self):
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(controls, text="Open", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(controls, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(controls, text="Reset", command=self.reset_image).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(controls, text="Algorithm:").pack(side=tk.LEFT)
        algo_menu = tk.OptionMenu(controls, self.algorithm, *sorted(ALGORITHMS))
        algo_menu.config(width=8)
        algo_menu.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(controls, text="Step seams:").pack(side=tk.LEFT)
        tk.Spinbox(controls, from_=1, to=500, width=5, textvariable=self.step_size).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(controls, text="Delay ms:").pack(side=tk.LEFT)
        tk.Spinbox(controls, from_=0, to=2000, width=6, textvariable=self.delay_ms).pack(side=tk.LEFT, padx=(2, 8))

        tk.Checkbutton(controls, text="Step-by-step", variable=self.show_steps).pack(side=tk.LEFT)
        tk.Button(controls, text="Remove", command=self.remove_seams).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(controls, text="Show All Seams", command=self.show_all_seams).pack(side=tk.LEFT)

        self.canvas = tk.Label(self.root, bd=1, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8, pady=(0, 8))

    def _set_status(self, text: str):
        self.status.set(text)
        self.root.update_idletasks()

    def _to_photo(self, image: np.ndarray) -> tk.PhotoImage:
        max_w, max_h = 1100, 700
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

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.original = load_image(path)
            self.current = self.original.copy()
            self.last_result = None
            self._render(self.current)
            self._set_status(f"Loaded {Path(path).name} ({self.current.shape[1]}x{self.current.shape[0]})")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def save_image(self):
        path = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            initialfile=Path(DEFAULT_OUTPUT).name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_image(self.current, path)
            self._set_status(f"Saved {path} ({self.current.shape[1]}x{self.current.shape[0]})")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def reset_image(self):
        self.current = self.original.copy()
        self.last_result = None
        self._render(self.current)
        self._set_status(f"Reset to {self.current.shape[1]}x{self.current.shape[0]}")

    def _step_callback(self, info):
        self._render(info["frame"])
        self._set_status(f"Step {info['step']}/{info['total']} | size {info['image'].shape[1]}x{info['image'].shape[0]}")
        self.root.update()
        delay = max(0, int(self.delay_ms.get()))
        if delay:
            self.root.after(delay)

    def remove_seams(self):
        n = max(1, int(self.step_size.get()))
        if self.current.shape[1] <= n:
            messagebox.showwarning("Cannot remove", "Not enough width to remove that many seams.")
            return
        algo = self.algorithm.get()
        try:
            carver = get_algorithm(algo)
            result = carver.run(
                self.current,
                num_seams=n,
                visualize=False,
                show_all_seams=True,
                step_callback=self._step_callback if self.show_steps.get() else None,
            )
            self.current = result.image
            self.last_result = result
            self._render(self.current)
            self._set_status(
                f"[{algo}] Removed {n} seam(s) -> {self.current.shape[1]}x{self.current.shape[0]} in {result.runtime_s:.3f}s"
            )
        except NotImplementedError as e:
            messagebox.showinfo("Not implemented", str(e))
            self._set_status(f"[{algo}] not implemented")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_all_seams(self):
        source = self.last_result
        if source is None or source.metadata.get("all_seams_image") is None:
            algo = self.algorithm.get()
            try:
                n = min(max(1, int(self.step_size.get())), self.current.shape[1] - 1)
                if n <= 0:
                    return
                carver = get_algorithm(algo)
                source = carver.run(self.current.copy(), num_seams=n, visualize=False, show_all_seams=True)
                self.last_result = source
            except NotImplementedError as e:
                messagebox.showinfo("Not implemented", str(e))
                self._set_status(f"[{algo}] not implemented")
                return
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
        overlay = source.metadata.get("all_seams_image")
        if overlay is None:
            return
        win = tk.Toplevel(self.root)
        win.title("All Seams (on original/base)")
        photo = self._to_photo(overlay)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)


def main():
    root = tk.Tk()
    SeamCarvingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
