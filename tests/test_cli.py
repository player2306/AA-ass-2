"""End-to-end tests of the command line (main.py) on a tiny generated image."""

import subprocess
import sys
from pathlib import Path

import cv2
import pytest

import main
from tests.conftest import rand_img

ROOT = Path(__file__).resolve().parents[1]
H, W = 14, 18


@pytest.fixture
def input_png(write_png):
    return write_png(rand_img(H, W, seed=11))


def cli(*args):
    return main.main([str(a) for a in args] + ["--no-show"])


def shape_of(path):
    image = cv2.imread(str(path))
    assert image is not None, f"{path} was not written"
    return image.shape[:2]


@pytest.mark.parametrize(
    "args, expected",
    [
        (["-a", "dp", "-n", 3], (H, W - 3)),
        (["-a", "greedy", "-n", 3, "-d", "both"], (H - 3, W - 3)),
        (["-a", "graphcut", "-n", 2, "-d", "horizontal"], (H - 2, W)),
        (["--width", 10, "--height", 9], (9, 10)),
    ],
    ids=["dp", "greedy-both", "graphcut-horizontal", "target-size"],
)
def test_carve_writes_resized_output(input_png, tmp_path, args, expected):
    out = tmp_path / "out" / "result.png"
    assert cli(input_png, *args, "-o", out, "--show-all-seams") == 0
    assert shape_of(out) == expected
    assert shape_of(out.with_name("result_all_seams.png")) == (H, W)


@pytest.mark.parametrize(
    "args",
    [["-n", W], ["-n", -1], ["--width", W + 1], ["-o", "{tmp}/x.notanimage"]],
    ids=["too-many-seams", "negative-seams", "target-too-big", "bad-output-extension"],
)
def test_errors_return_1(input_png, tmp_path, capsys, args):
    args = [str(a).replace("{tmp}", str(tmp_path)) for a in args]
    if "-o" not in args:
        args += ["-o", str(tmp_path / "result.png")]
    assert cli(input_png, *args) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_image_returns_error(tmp_path, capsys):
    assert cli(tmp_path / "missing.png") == 1
    assert "error:" in capsys.readouterr().err


def test_subprocess_smoke(input_png, tmp_path):
    out = tmp_path / "result.png"
    proc = subprocess.run(
        [sys.executable, "main.py", str(input_png), "-n", "2", "-o", str(out), "--no-show"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert shape_of(out) == (H, W - 2)
