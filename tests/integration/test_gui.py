"""Drive the Tkinter GUI programmatically (no clicks, dialogs patched out)."""

import tkinter as tk

import cv2
import numpy as np
import pytest

import gui
from tests.conftest import rand_img

H, W = 20, 24


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes instead of showing them; file dialogs return a preset path."""
    shown = []
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(gui.messagebox, name, lambda title, msg, _n=name: shown.append((_n, title, msg)))
    paths = {"open": "", "save": ""}
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **k: paths["open"])
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: paths["save"])
    return shown, paths


@pytest.fixture(scope="module")
def tk_root():
    # One Tk interpreter for the whole module: creating many tk.Tk() in a row is
    # flaky on Windows (intermittent "invalid command name tcl_findLibrary").
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display / broken Tcl install
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


def carve(app, algorithm="dp", direction="vertical", n=3, **settings):
    app.algorithm.set(algorithm)
    app._on_algorithm_change()
    app.direction.set(direction)
    app.num_seams.set(n)
    for key, value in settings.items():
        getattr(app, key).set(value)
    app.remove_seams()


@pytest.mark.parametrize("algorithm", ["dp", "greedy", "graphcut"])
@pytest.mark.parametrize(
    "direction, expected",
    [("vertical", (H, W - 3)), ("horizontal", (H - 3, W)), ("both", (H - 3, W - 3))],
)
def test_remove_seams_every_algorithm_and_direction(app, dialogs, algorithm, direction, expected):
    carve(app, algorithm, direction, 3)
    assert dialogs[0] == []
    assert app.current.shape[:2] == expected
    assert app.last_result.metadata["all_seams_image"].shape == (H, W, 3)
    assert "removed" in app.status.get()


def test_algorithm_options_are_passed_to_the_carver(app):
    app.algorithm.set("greedy")
    app.multi_start.set(True)
    assert app.build_run_kwargs()["multi_start"] is True
    app.algorithm.set("graphcut")
    app.energy.set("forward")
    app.method.set("edmonds_karp")
    kwargs = app.build_run_kwargs()
    assert (kwargs["energy"], kwargs["method"]) == ("forward", "edmonds_karp")
    assert "multi_start" not in kwargs
    app.algorithm.set("dp")
    assert set(app.build_run_kwargs()) == {"direction", "visualize", "show_all_seams"}


def test_options_panel_matches_algorithm(app):
    app.algorithm.set("greedy")
    app._on_algorithm_change()
    assert app.greedy_opts.winfo_manager() and not app.graphcut_opts.winfo_manager()
    app.algorithm.set("graphcut")
    app._on_algorithm_change()
    assert app.graphcut_opts.winfo_manager() and not app.greedy_opts.winfo_manager()
    app.algorithm.set("dp")
    app._on_algorithm_change()
    assert not app.greedy_opts.winfo_manager() and not app.graphcut_opts.winfo_manager()


def test_step_by_step_runs_callback(app):
    carve(app, "dp", "both", 2, show_steps=True, delay_ms=0)
    assert app.current.shape[:2] == (H - 2, W - 2)
    assert not app.busy


@pytest.mark.parametrize(
    "direction, n",
    [("vertical", W), ("horizontal", H), ("both", H), ("vertical", 0), ("vertical", -2)],
)
def test_out_of_range_seams_warn_and_keep_image(app, dialogs, direction, n):
    carve(app, "dp", direction, n)
    assert app.current.shape[:2] == (H, W)
    assert dialogs[0] and dialogs[0][-1][0] == "showwarning"


def test_non_numeric_seams_warns(app, dialogs):
    app.num_seams.set("abc")  # IntVar accepts the string but .get() fails
    app.remove_seams()
    assert dialogs[0][-1][0] == "showwarning"
    assert app.current.shape[:2] == (H, W)


def test_remove_max_seams_leaves_one_column(app):
    carve(app, "dp", "vertical", W - 1)
    assert app.current.shape[:2] == (H, 1)


def test_undo_and_reset(app):
    carve(app, n=2)
    carve(app, n=3)
    assert app.current.shape[1] == W - 5
    app.undo()
    assert app.current.shape[1] == W - 2
    app.undo()
    assert app.current.shape[1] == W
    app.undo()  # nothing left: no error
    carve(app, n=4)
    app.reset_image()
    np.testing.assert_array_equal(app.current, app.original)
    assert app.history == []


def test_apply_scale_shrinks_base_image(app):
    app.scale_pct.set(50)
    app.apply_scale()
    assert app.current.shape[:2] == (H // 2, W // 2)
    carve(app, n=2)
    app.reset_image()
    assert app.current.shape[:2] == (H // 2, W // 2)  # reset returns to the scaled base
    app.scale_pct.set(100)
    app.apply_scale()
    assert app.current.shape[:2] == (H, W)


def test_compare_view_renders(app):
    carve(app, n=3)
    app.compare.set(True)
    app._refresh()
    assert app.photo.width() == 2 * W + 16 or app.photo.width() <= gui.MAX_VIEW_W


def test_save_image_and_overlay(app, dialogs, tmp_path):
    _, paths = dialogs
    carve(app, n=3)
    paths["save"] = str(tmp_path / "out" / "carved.png")
    app.save_image()
    assert cv2.imread(paths["save"]).shape[:2] == (H, W - 3)
    paths["save"] = str(tmp_path / "seams.png")
    app.save_overlay()
    assert cv2.imread(paths["save"]).shape[:2] == (H, W)


def test_save_overlay_before_any_run_informs(app, dialogs):
    app.save_overlay()
    assert dialogs[0][-1][0] == "showinfo"


def test_save_bad_extension_shows_error(app, dialogs, tmp_path):
    dialogs[1]["save"] = str(tmp_path / "x.notanimage")
    app.save_image()
    assert dialogs[0][-1][0] == "showerror"


def test_cancelled_dialogs_do_nothing(app, dialogs):
    before = app.current.copy()
    app.open_image()
    app.save_image()
    np.testing.assert_array_equal(app.current, before)
    assert dialogs[0] == []


def test_open_image(app, dialogs, write_png):
    dialogs[1]["open"] = str(write_png(rand_img(9, 13, seed=4), "other.png"))
    carve(app, n=2)
    app.open_image()
    assert app.current.shape[:2] == (9, 13)
    assert app.history == [] and app.last_result is None
    assert "other.png" in app.status.get()


def test_open_bad_file_shows_error(app, dialogs, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    dialogs[1]["open"] = str(bad)
    app.open_image()
    assert dialogs[0][-1][0] == "showerror"
    assert app.current.shape[:2] == (H, W)


def test_show_all_seams_window(app, dialogs):
    app.show_all_seams()
    assert dialogs[0][-1][0] == "showinfo"  # nothing removed yet
    carve(app, n=2)
    app.show_all_seams()
    assert any(isinstance(w, tk.Toplevel) for w in app.root.winfo_children())


def test_grayscale_image_displays(app):
    app.set_image(rand_img(10, 12, channels=0), "gray.png")
    carve(app, "dp", "vertical", 2)
    assert app.current.shape == (10, 10)
