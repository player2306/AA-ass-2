"""Drive the Tkinter GUI from code (no clicks; dialogs are recorded instead of shown)."""

import tkinter as tk

import cv2
import pytest

import gui
from tests.conftest import rand_img

H, W = 20, 24


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes; file dialogs return preset paths; Yes/No answers `answer`."""
    state = {"shown": [], "save": "", "answer": False}
    for kind in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(gui.messagebox, kind, lambda t, m, _k=kind: state["shown"].append((_k, m)))

    def askyesno(title, msg, **kwargs):
        state["shown"].append(("askyesno", msg))
        return state["answer"]

    monkeypatch.setattr(gui.messagebox, "askyesno", askyesno)
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: state["save"])
    return state


@pytest.fixture(scope="module")
def tk_root():
    # one Tk interpreter per module: creating many in a row is flaky on Windows
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk not available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root, dialogs):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    yield gui.SeamCarvingGUI(window, image=rand_img(H, W, seed=1), name="test.png")
    window.destroy()


def remove(app, algorithm="dp", direction="vertical", n=3):
    app.algorithm.set(algorithm)
    app._on_algorithm_change()
    app.direction.set(direction)
    app.num_seams.set(n)
    app.remove_seams()


@pytest.mark.parametrize("algorithm", ["dp", "greedy", "graphcut"])
def test_remove_seams(app, dialogs, algorithm):
    remove(app, algorithm, "both", 3)
    assert dialogs["shown"] == []
    assert app.current.shape[:2] == (H - 3, W - 3)


def test_carve_to_target_size(app, dialogs):
    app.target_w.set(15)
    app.target_h.set(12)
    app.carve_to_target()
    assert app.current.shape[:2] == (12, 15)
    app.target_w.set(W + 1)  # bigger than the image: warn, keep image
    app.carve_to_target()
    assert dialogs["shown"][-1][0] == "showwarning"
    assert app.current.shape[:2] == (12, 15)


def test_undo_and_reset(app):
    remove(app, n=2)
    remove(app, n=3)
    app.undo()
    assert app.current.shape[:2] == (H, W - 2)
    app.reset_image()
    assert app.current.shape[:2] == (H, W)
    assert (app.target_w.get(), app.target_h.get()) == (W, H)  # target fields follow the image


@pytest.mark.parametrize("n", [W, 0, "abc"])
def test_invalid_seam_count_warns(app, dialogs, n):
    app.num_seams.set(n)
    app.remove_seams()
    assert dialogs["shown"][-1][0] == "showwarning"
    assert app.current.shape[:2] == (H, W)


def test_slow_algorithm_warning(app, dialogs, monkeypatch):
    # pretend graph cut takes 100 s per seam on this image
    monkeypatch.setitem(gui.SLOW_ALGORITHMS, "graphcut", {"ref_pixels": H * W, "ref_seconds": 100.0, "exponent": 1.0})
    remove(app, "graphcut", "vertical", 2)  # answer No
    assert dialogs["shown"][-1][0] == "askyesno"
    assert app.current.shape[:2] == (H, W)
    dialogs["answer"] = True
    remove(app, "graphcut", "vertical", 2)
    assert app.current.shape[:2] == (H, W - 2)
    remove(app, "dp", "vertical", 2)  # fast algorithms never ask
    assert sum(kind == "askyesno" for kind, _ in dialogs["shown"]) == 2


def test_save_image_and_overlay(app, dialogs, tmp_path):
    remove(app, n=3)
    dialogs["save"] = str(tmp_path / "carved.png")
    app.save_image()
    assert cv2.imread(dialogs["save"]).shape[:2] == (H, W - 3)
    dialogs["save"] = str(tmp_path / "seams.png")
    app.save_overlay()
    assert cv2.imread(dialogs["save"]).shape[:2] == (H, W)
