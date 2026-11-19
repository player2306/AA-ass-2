"""End-to-end tests of the command line (main.py) on tiny generated images."""

import subprocess
import sys
from pathlib import Path

import cv2
import pytest

import main
from tests.conftest import rand_img

ROOT = Path(__file__).resolve().parents[2]
H, W = 14, 18


@pytest.fixture
def input_png(write_png):
    return write_png(rand_img(H, W, seed=11))


@pytest.fixture
def out_png(tmp_path):
    return tmp_path / "out" / "result.png"


def cli(*args):
    return main.main([str(a) for a in args] + ["--no-show"])


def shape_of(path):
    image = cv2.imread(str(path))
    assert image is not None, f"{path} was not written"
    return image.shape[:2]


@pytest.mark.parametrize(
    "direction, expected",
    [("vertical", (H, W - 3)), ("horizontal", (H - 3, W)), ("both", (H - 3, W - 3))],
)
@pytest.mark.parametrize("algorithm", ["dp", "greedy"])
def test_carve_writes_resized_output(input_png, out_png, algorithm, direction, expected):
    assert cli(input_png, "-a", algorithm, "-n", 3, "-d", direction, "-o", out_png) == 0
    assert shape_of(out_png) == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction, expected",
    [("vertical", (H, W - 2)), ("horizontal", (H - 2, W)), ("both", (H - 2, W - 2))],
)
def test_graphcut_through_cli(input_png, out_png, direction, expected):
    assert cli(input_png, "-a", "graphcut", "-n", 2, "-d", direction, "-o", out_png) == 0
    assert shape_of(out_png) == expected


def test_zero_seams_keeps_size(input_png, out_png):
    assert cli(input_png, "-n", 0, "-o", out_png) == 0
    assert shape_of(out_png) == (H, W)


def test_max_seams_leaves_one_column(input_png, out_png):
    assert cli(input_png, "-n", W - 1, "-o", out_png) == 0
    assert shape_of(out_png) == (H, 1)


def test_show_all_seams_writes_default_overlay_path(input_png, out_png):
    assert cli(input_png, "-n", 2, "-o", out_png, "--show-all-seams") == 0
    assert shape_of(out_png.with_name("result_all_seams.png")) == (H, W)


def test_all_seams_output_custom_path(input_png, out_png, tmp_path):
    custom = tmp_path / "overlays" / "seams.png"
    assert cli(input_png, "-n", 2, "-o", out_png, "--all-seams-output", custom) == 0
    assert shape_of(custom) == (H, W)
    assert not out_png.with_name("result_all_seams.png").exists()


def test_preview_does_not_write_output(input_png, out_png):
    assert cli(input_png, "--preview", "-o", out_png) == 0
    assert not out_png.exists()


def test_missing_image_returns_error(tmp_path, out_png, capsys):
    assert cli(tmp_path / "missing.png", "-o", out_png) == 1
    assert "error:" in capsys.readouterr().err
    assert not out_png.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["-n", W],
        ["-n", 999],
        ["-n", -1],
        ["-n", H, "-d", "horizontal"],
        ["-n", H, "-d", "both"],  # fits the width (18) but not the height (14)
    ],
    ids=["n=width", "n-huge", "n-negative", "n=height-horizontal", "n=height-both"],
)
def test_bad_seam_count_returns_error(input_png, out_png, capsys, args):
    assert cli(input_png, *args, "-o", out_png) == 1
    assert "error:" in capsys.readouterr().err
    assert not out_png.exists()


def test_grayscale_input_png(write_png, out_png):
    # load_image always returns 3 channels, so a grey PNG must work end to end
    path = write_png(rand_img(H, W, seed=2, channels=0), "gray.png")
    assert cli(path, "-n", 2, "-o", out_png) == 0
    assert shape_of(out_png) == (H, W - 2)


def test_unwritable_output_extension_reports_error(input_png, tmp_path, capsys):
    assert cli(input_png, "-n", 1, "-o", tmp_path / "result.notanimage") == 1
    assert "could not write" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["-a", "foo"], ["-d", "diagonal"], ["-n", "abc"]])
def test_invalid_arguments_exit_with_usage_error(input_png, args):
    with pytest.raises(SystemExit) as exc:
        cli(input_png, *args)
    assert exc.value.code == 2


def test_jpeg_output(input_png, tmp_path):
    out = tmp_path / "result.jpg"
    assert cli(input_png, "-n", 1, "-o", out) == 0
    assert shape_of(out) == (H, W - 1)


def test_subprocess_smoke(input_png, out_png):
    proc = subprocess.run(
        [sys.executable, "main.py", str(input_png), "-n", "2", "-o", str(out_png), "--no-show"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Saved" in proc.stdout
    assert shape_of(out_png) == (H, W - 2)


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--width", 10, "--height", 9], (9, 10)),
        (["--width", 5], (H, 5)),
        (["--height", 3], (3, W)),
        (["--width", W, "--height", H], (H, W)),
        (["--width", 1, "--height", 1], (1, 1)),
    ],
    ids=["both", "width-only", "height-only", "unchanged", "one-pixel"],
)
@pytest.mark.parametrize("algorithm", ["dp", "greedy"])
def test_carve_to_target_size(input_png, out_png, algorithm, args, expected):
    assert cli(input_png, "-a", algorithm, *args, "-o", out_png) == 0
    assert shape_of(out_png) == expected


def test_target_size_overrides_seam_count(input_png, out_png):
    assert cli(input_png, "-n", 2, "-d", "both", "--width", 10, "-o", out_png) == 0
    assert shape_of(out_png) == (H, 10)


def test_target_size_with_all_seams_overlay(input_png, out_png):
    assert cli(input_png, "--width", 12, "--height", 10, "-o", out_png, "--show-all-seams") == 0
    assert shape_of(out_png.with_name("result_all_seams.png")) == (H, W)


@pytest.mark.slow
def test_graphcut_to_target_size(input_png, out_png):
    assert cli(input_png, "-a", "graphcut", "--width", 16, "--height", 12, "-o", out_png) == 0
    assert shape_of(out_png) == (12, 16)


@pytest.mark.parametrize(
    "args",
    [["--width", W + 1], ["--height", H + 1], ["--width", 0], ["--height", -5]],
    ids=["wider", "taller", "zero-width", "negative-height"],
)
def test_unreachable_target_size_returns_error(input_png, out_png, capsys, args):
    assert cli(input_png, *args, "-o", out_png) == 1
    assert "only shrink" in capsys.readouterr().err
    assert not out_png.exists()
