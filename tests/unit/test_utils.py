"""Unit tests for the shared helpers in src/utils and src/seam_utils.py."""

import time

import numpy as np
import pytest

from src.seam_utils import compute_energy, seam_energy
from src.utils import draw_seam, draw_seams, remove_vertical_seam, sobel_edge_map
from src.utils.filters import sobel_gradients
from src.utils.timing import TimerResult, timer
from tests.conftest import rand_img


# ---------------------------------------------------------------- remove_vertical_seam


@pytest.mark.parametrize("channels", [3, 0], ids=["color", "gray"])
def test_remove_seam_shrinks_width_by_one_and_keeps_dtype(channels):
    image = rand_img(6, 5, seed=1, channels=channels)
    seam = np.array([0, 1, 2, 3, 4, 4])
    out = remove_vertical_seam(image, seam)
    assert out.shape[:2] == (6, 4)
    assert out.ndim == image.ndim
    assert out.dtype == image.dtype
    for r, c in enumerate(seam):
        np.testing.assert_array_equal(out[r], np.delete(image[r], c, axis=0))


@pytest.mark.parametrize("col", [0, -1], ids=["first-column", "last-column"])
def test_remove_seam_at_image_border(col):
    image = rand_img(4, 5, seed=2)
    c = col % 5
    out = remove_vertical_seam(image, np.full(4, c))
    np.testing.assert_array_equal(out, np.delete(image, c, axis=1))


def test_remove_seam_from_width_one_image_gives_empty_width():
    out = remove_vertical_seam(rand_img(3, 1), np.zeros(3, dtype=int))
    assert out.shape == (3, 0, 3)


def test_remove_seam_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        remove_vertical_seam(rand_img(4, 5), np.zeros(3, dtype=int))


@pytest.mark.parametrize("bad", [-1, 5, 100])
def test_remove_seam_out_of_bounds_raises(bad):
    seam = np.array([0, 0, bad, 0])
    with pytest.raises(ValueError, match="out of bounds"):
        remove_vertical_seam(rand_img(4, 5), seam)


def test_remove_seam_does_not_mutate_input():
    image = rand_img(4, 5)
    before = image.copy()
    remove_vertical_seam(image, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(image, before)


# ---------------------------------------------------------------- draw_seam(s)


def test_draw_seam_returns_copy_and_marks_pixels():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    out = draw_seam(image, np.array([0, 1, 2]), color=(1, 2, 3))
    assert not image.any()  # input untouched
    for r, c in enumerate([0, 1, 2]):
        assert tuple(out[r, c]) == (1, 2, 3)
    assert (out != 0).any(axis=2).sum() == 3


def test_draw_seam_ignores_out_of_range_indices():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    out = draw_seam(image, np.array([-1, 99, 1, 2, 2]))  # also longer than the height
    assert (out != 0).any(axis=2).sum() == 1  # only row 2, col 1 is valid
    assert out[2, 1].any()


def test_draw_seams_empty_list_is_unchanged_copy():
    image = rand_img(3, 4)
    out = draw_seams(image, [])
    np.testing.assert_array_equal(out, image)
    assert out is not image


# ---------------------------------------------------------------- energy / filters


def test_energy_of_flat_image_is_zero_everywhere_including_borders():
    energy = sobel_edge_map(np.full((5, 6, 3), 123, dtype=np.uint8))
    assert energy.shape == (5, 6)
    assert energy.dtype == np.float32
    assert not energy.any()


def test_energy_of_vertical_step_is_only_near_the_edge():
    image = np.zeros((6, 8), dtype=np.uint8)
    image[:, 4:] = 255
    energy = sobel_edge_map(image)
    assert energy[:, 3:5].min() > 0
    assert not energy[:, :2].any()
    assert not energy[:, 6:].any()


def test_energy_gray_and_color_inputs_agree():
    gray = rand_img(5, 7, channels=0)
    color = np.dstack([gray, gray, gray])
    np.testing.assert_allclose(sobel_edge_map(gray), sobel_edge_map(color), atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (6, 1)])
def test_energy_on_degenerate_sizes(shape):
    energy = sobel_edge_map(rand_img(*shape, channels=0))
    assert energy.shape == shape
    assert np.isfinite(energy).all()


def test_sobel_gradients_signs():
    ramp = np.tile(np.arange(6, dtype=np.uint8) * 10, (4, 1))  # brightens to the right
    gx, gy = sobel_gradients(ramp)
    assert (gx[:, 1:-1] > 0).all()
    assert not gy.any()


def test_compute_energy_matches_shared_sobel():
    image = rand_img(6, 7)
    np.testing.assert_array_equal(compute_energy(image), sobel_edge_map(image))


def test_seam_energy_sums_exactly_in_float64():
    energy = np.full((1000, 2), 0.1, dtype=np.float32)
    total = seam_energy(energy, np.zeros(1000, dtype=int))
    assert isinstance(total, float)
    assert total == pytest.approx(1000 * float(np.float32(0.1)), rel=1e-12)


# ---------------------------------------------------------------- timer


def test_timer_measures_elapsed_time():
    result = TimerResult()
    with timer(result):
        time.sleep(0.01)
    assert result.elapsed_s >= 0.005


def test_timer_without_argument_yields_result():
    with timer() as result:
        pass
    assert result.elapsed_s >= 0


def test_timer_records_time_even_when_body_raises():
    result = TimerResult()
    with pytest.raises(RuntimeError):
        with timer(result):
            raise RuntimeError("boom")
    assert result.elapsed_s > 0


def test_draw_seam_on_grayscale_returns_bgr_with_colored_seam():
    gray = np.full((3, 4), 50, dtype=np.uint8)
    out = draw_seam(gray, np.array([0, 1, 2]), color=(0, 0, 255))
    assert out.shape == (3, 4, 3)
    assert tuple(out[1, 1]) == (0, 0, 255)
    assert tuple(out[0, 3]) == (50, 50, 50)
    assert gray.ndim == 2  # input untouched
