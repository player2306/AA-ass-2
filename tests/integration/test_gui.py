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
    paths = {"open": "", "save": "", "slow_answer": False}

    def askyesno(title, msg, **kwargs):
        shown.append(("askyesno", title, msg))
        return paths["slow_answer"]

    monkeypatch.setattr(gui.messagebox, "askyesno", askyesno)
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


# ---------------------------------------------------------------- slow-algorithm warning


def test_estimate_matches_reference_timings():
    assert gui.estimate_seconds("graphcut", 97, 143, 1, "vertical") == pytest.approx(1.0)
    assert gui.estimate_seconds("graphcut", 194, 286, 1, "vertical") == pytest.approx(9.0, rel=0.05)


def test_estimate_is_zero_for_fast_algorithms_and_no_seams():
    assert gui.estimate_seconds("dp", 968, 1428, 500, "both") == 0
    assert gui.estimate_seconds("greedy", 968, 1428, 500, "both") == 0
    assert gui.estimate_seconds("graphcut", 968, 1428, 0, "both") == 0


def test_estimate_grows_with_seams_and_direction():
    one = gui.estimate_seconds("graphcut", 100, 120, 5, "vertical")
    assert gui.estimate_seconds("graphcut", 100, 120, 10, "vertical") > one
    assert gui.estimate_seconds("graphcut", 100, 120, 5, "both") > one
    # each removed seam makes the image smaller, so later seams are cheaper
    assert gui.estimate_seconds("graphcut", 100, 120, 5, "vertical") < 5 * gui.estimate_seconds(
        "graphcut", 100, 120, 1, "vertical"
    )


def test_suggested_scale_fits_target_and_none_when_impossible():
    pct = gui.suggest_scale_pct("graphcut", 968, 1428, 5, "both")
    assert pct is not None and 0 < pct < 100
    h, w = round(968 * pct / 100), round(1428 * pct / 100)
    assert gui.estimate_seconds("graphcut", h, w, 5, "both") <= gui.SLOW_WARNING_SECONDS
    assert gui.suggest_scale_pct("graphcut", 968, 1428, 900, "both") is None  # too many seams to shrink


@pytest.mark.parametrize("seconds, text", [(5, "5 seconds"), (600, "10 minutes"), (7200, "2.0 hours")])
def test_format_duration(seconds, text):
    assert gui.format_duration(seconds) == text


@pytest.fixture
def slow_graphcut(monkeypatch):
    """Pretend graph cut takes 100 s per seam on the tiny test image."""
    monkeypatch.setitem(gui.SLOW_ALGORITHMS, "graphcut", {"ref_pixels": H * W, "ref_seconds": 100.0, "exponent": 1.0})


def test_slow_run_warns_and_cancel_keeps_image(app, dialogs, slow_graphcut):
    shown, paths = dialogs
    paths["slow_answer"] = False
    carve(app, "graphcut", "both", 2)
    kind, title, msg = shown[-1]
    assert kind == "askyesno" and title == "Slow algorithm"
    assert "graphcut is slow" in msg and "2 vertical + 2 horizontal" in msg and "minutes" in msg
    assert app.current.shape[:2] == (H, W)
    assert app.history == []
    assert "cancelled" in app.status.get()


def test_slow_run_confirmed_carves(app, dialogs, slow_graphcut):
    dialogs[1]["slow_answer"] = True
    carve(app, "graphcut", "vertical", 2)
    assert dialogs[0][-1][0] == "askyesno"
    assert app.current.shape[:2] == (H, W - 2)


def test_slow_warning_suggests_scale_based_on_opened_image(app, dialogs, monkeypatch):
    big = np.zeros((400, 600, 3), np.uint8)
    app.set_image(big, "big.png")
    monkeypatch.setitem(gui.SLOW_ALGORITHMS, "graphcut", {"ref_pixels": 100 * 100, "ref_seconds": 1.0, "exponent": 1.5})
    app.current = big  # answer defaults to "No", so nothing actually runs
    carve(app, "graphcut", "vertical", 3)
    msg = dialogs[0][-1][2]
    assert "Scale %" in msg and "Apply scale" in msg
    pct = gui.suggest_scale_pct("graphcut", 400, 600, 3, "vertical")
    assert f"to {pct} " in msg and f"{round(600 * pct / 100)}x{round(400 * pct / 100)}" in msg


@pytest.mark.parametrize("algorithm", ["dp", "greedy"])
def test_fast_algorithms_never_warn(app, dialogs, slow_graphcut, algorithm):
    carve(app, algorithm, "both", 3)
    assert dialogs[0] == []
    assert app.current.shape[:2] == (H - 3, W - 3)


def test_small_graphcut_run_does_not_warn(app, dialogs):
    carve(app, "graphcut", "vertical", 2)  # real estimate on 20x24 is a fraction of a second
    assert dialogs[0] == []
    assert app.current.shape[:2] == (H, W - 2)


# ---------------------------------------------------------------- carve to a target size


def to_target(app, w, h, algorithm="dp"):
    app.algorithm.set(algorithm)
    app._on_algorithm_change()
    app.target_w.set(w)
    app.target_h.set(h)
    app.carve_to_target()


def test_target_fields_start_at_image_size(app):
    assert (app.target_w.get(), app.target_h.get()) == (W, H)


@pytest.mark.parametrize("algorithm", ["dp", "greedy", "graphcut"])
@pytest.mark.parametrize("target", [(18, 15), (W, 10), (10, H), (1, 1)])
def test_carve_to_target(app, dialogs, algorithm, target):
    to_target(app, *target, algorithm=algorithm)
    assert dialogs[0] == []
    assert app.current.shape[:2] == (target[1], target[0])
    assert f"-> {target[0]}x{target[1]}" in app.status.get()


def test_target_fields_follow_image_after_carving_and_undo(app):
    carve(app, "dp", "vertical", 4)
    assert (app.target_w.get(), app.target_h.get()) == (W - 4, H)
    to_target(app, 15, 12)
    assert (app.target_w.get(), app.target_h.get()) == (15, 12)
    app.undo()
    assert (app.target_w.get(), app.target_h.get()) == (W - 4, H)
    app.reset_image()
    assert (app.target_w.get(), app.target_h.get()) == (W, H)


def test_typed_target_survives_view_toggle(app):
    app.target_w.set(9)
    app.compare.set(True)
    app._refresh()
    assert app.target_w.get() == 9


def test_carve_to_target_is_one_undo_step(app):
    to_target(app, 14, 11)
    assert len(app.history) == 1
    app.undo()
    assert app.current.shape[:2] == (H, W)


@pytest.mark.parametrize("target", [(W + 1, H), (W, H + 5), (0, 5), (5, 0), (-3, 4)])
def test_unreachable_target_warns_and_keeps_image(app, dialogs, target):
    to_target(app, *target)
    kind, title, msg = dialogs[0][-1]
    assert kind == "showwarning" and "only make the image smaller" in msg
    assert app.current.shape[:2] == (H, W)


def test_non_numeric_target_warns(app, dialogs):
    app.target_w.set("wide")
    app.carve_to_target()
    assert dialogs[0][-1][0] == "showwarning"
    assert app.current.shape[:2] == (H, W)


def test_unchanged_target_does_nothing(app, dialogs):
    to_target(app, W, H)
    assert dialogs[0] == []
    assert app.history == []
    assert "Already" in app.status.get()


def test_target_overlay_has_original_size(app):
    to_target(app, 20, 16)
    assert app.last_result.metadata["all_seams_image"].shape == (H, W, 3)


def test_slow_target_run_warns_with_target_tip(app, dialogs, slow_graphcut):
    to_target(app, 20, 18, algorithm="graphcut")  # the recorded answer defaults to No
    kind, title, msg = dialogs[0][-1]
    assert kind == "askyesno" and "4 vertical + 2 horizontal" in msg
    assert "closer to the current size" in msg
    assert app.current.shape[:2] == (H, W)


def test_slow_target_run_confirmed(app, dialogs, slow_graphcut):
    dialogs[1]["slow_answer"] = True
    to_target(app, 21, H, algorithm="graphcut")
    assert app.current.shape[:2] == (H, 21)


def test_plan_seconds_matches_direction_estimate():
    both = gui.estimate_seconds("graphcut", 100, 120, 5, "both")
    assert gui.plan_seconds("graphcut", 100, 120, 5, 5) == pytest.approx(both)
    assert gui.plan_seconds("graphcut", 100, 120, 0, 0) == 0
    assert gui.plan_seconds("dp", 100, 120, 50, 50) == 0


@pytest.mark.parametrize(
    "nv, nh, text", [(3, 0, "3 vertical"), (0, 2, "2 horizontal"), (3, 2, "3 vertical + 2 horizontal")]
)
def test_describe_seams(nv, nh, text):
    assert gui.describe_seams(nv, nh) == text
