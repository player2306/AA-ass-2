"""Unit tests for image IO, the display helper and the algorithm registry."""

import numpy as np
import pytest

from src.algorithms import ALGORITHMS, get_algorithm
from src.algorithms.base import SeamCarver
from src.display import fit_to_window
from src.image_io import load_image, save_image
from tests.conftest import rand_img


# ---------------------------------------------------------------- load_image


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(tmp_path / "nope.png")


def test_load_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path)


@pytest.mark.parametrize("content", [b"", b"this is not an image"], ids=["empty", "text"])
def test_load_undecodable_file_raises(tmp_path, content):
    path = tmp_path / "fake.png"
    path.write_bytes(content)
    with pytest.raises(FileNotFoundError, match="could not read"):
        load_image(path)


def test_load_grayscale_png_returns_3_channel_bgr(write_png):
    path = write_png(rand_img(4, 5, channels=0))
    image = load_image(str(path))  # str paths work too
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8


# ---------------------------------------------------------------- save_image


def test_save_creates_nested_dirs_and_png_roundtrips(tmp_path):
    image = rand_img(7, 9)
    out = save_image(image, tmp_path / "a" / "b" / "c" / "out.png")
    assert out.is_file()
    np.testing.assert_array_equal(load_image(out), image)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: save_image promises IOError, but cv2.imwrite raises cv2.error for an unknown extension",
)
def test_save_unknown_extension_raises_ioerror(tmp_path):
    with pytest.raises(IOError):
        save_image(rand_img(3, 3), tmp_path / "out.notanimage")


def test_save_unknown_extension_raises_something(tmp_path):
    with pytest.raises(Exception):
        save_image(rand_img(3, 3), tmp_path / "out.notanimage")


# ---------------------------------------------------------------- fit_to_window


def test_fit_small_image_is_returned_untouched():
    image = rand_img(10, 10)
    assert fit_to_window(image, 100, 100) is image


def test_fit_exact_size_is_not_resized():
    image = rand_img(100, 200)
    assert fit_to_window(image, 200, 100) is image


def test_fit_large_image_scales_down_keeping_aspect_ratio():
    out = fit_to_window(np.zeros((400, 1000, 3), np.uint8), max_width=500, max_height=500)
    assert out.shape[:2] == (200, 500)


def test_fit_extreme_aspect_ratio_never_reaches_zero_pixels():
    out = fit_to_window(np.zeros((1, 10000, 3), np.uint8), max_width=100, max_height=100)
    assert out.shape[0] >= 1 and out.shape[1] == 100


# ---------------------------------------------------------------- registry


def test_registry_has_all_three_algorithms():
    assert set(ALGORITHMS) == {"dp", "greedy", "graphcut"}


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_get_algorithm_returns_fresh_instance(name):
    a, b = get_algorithm(name), get_algorithm(name)
    assert isinstance(a, SeamCarver)
    assert a.name == name
    assert a is not b


@pytest.mark.parametrize("name", ["", "DP", "foo"])
def test_get_unknown_algorithm_lists_choices(name):
    with pytest.raises(ValueError) as exc:
        get_algorithm(name)
    for valid in ALGORITHMS:
        assert valid in str(exc.value)
