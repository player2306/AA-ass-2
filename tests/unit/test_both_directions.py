"""Unit tests for direction='both' (src/algorithms/both_directions.py)."""

import numpy as np
import pytest

from src.algorithms import get_algorithm
from src.algorithms.both_directions import _delete, _seam_pixels, carve_both_directions
from tests.conftest import rand_img

BOTH_CARVERS = ["dp", "greedy"]


@pytest.fixture
def green_image():
    """Image with no pure-red pixels, so the red overlay can be counted exactly."""
    image = np.zeros((9, 11, 3), dtype=np.uint8)
    image[:, :, 1] = rand_img(9, 11, seed=5, channels=0)
    return image


@pytest.mark.parametrize("name", BOTH_CARVERS)
@pytest.mark.parametrize("n", [1, 4, 8])
def test_overlay_marks_every_removed_pixel(name, green_image, n):
    h, w = green_image.shape[:2]
    result = get_algorithm(name).run(green_image, num_seams=n, direction="both", show_all_seams=True)
    assert result.image.shape == (h - n, w - n, 3)
    red = np.all(result.metadata["all_seams_image"] == (0, 0, 255), axis=2)
    assert red.sum() == h * w - (h - n) * (w - n)
    assert result.metadata["seams_removed"] == 2 * n
    assert result.metadata["direction"] == "both"


@pytest.mark.parametrize("name", BOTH_CARVERS)
def test_seams_are_listed_vertical_then_horizontal(name, green_image):
    seams = get_algorithm(name).run(green_image, num_seams=3, direction="both").metadata["seams"]
    assert [d for d, _ in seams] == ["vertical"] * 3 + ["horizontal"] * 3


@pytest.mark.parametrize("name", BOTH_CARVERS)
def test_callback_numbers_steps_across_both_batches(name, green_image):
    calls = []
    get_algorithm(name).run(green_image, num_seams=3, direction="both", step_callback=calls.append)
    assert [c["step"] for c in calls] == [1, 2, 3, 4, 5, 6]
    assert [c["direction"] for c in calls] == ["vertical"] * 3 + ["horizontal"] * 3
    assert all(c["total"] == 6 for c in calls)


@pytest.mark.parametrize("name", BOTH_CARVERS)
@pytest.mark.parametrize("shape, n", [((9, 11), 9), ((11, 9), 9), ((5, 20), 6), ((20, 5), 5)])
def test_n_must_be_smaller_than_both_sides(name, shape, n):
    with pytest.raises(ValueError):
        get_algorithm(name).run(rand_img(*shape), num_seams=n, direction="both")


@pytest.mark.parametrize("name", BOTH_CARVERS)
def test_negative_n_raises(name, green_image):
    with pytest.raises(ValueError):
        get_algorithm(name).run(green_image, num_seams=-1, direction="both")


@pytest.mark.parametrize("name", BOTH_CARVERS)
def test_zero_seams_is_identity(name, green_image):
    result = get_algorithm(name).run(green_image, num_seams=0, direction="both")
    np.testing.assert_array_equal(result.image, green_image)


def test_seam_energies_only_when_algorithm_reports_them(green_image):
    greedy = get_algorithm("greedy").run(green_image, num_seams=2, direction="both")
    assert len(greedy.metadata["seam_energies"]) == 4
    dp = get_algorithm("dp").run(green_image, num_seams=2, direction="both")
    assert "seam_energies" not in dp.metadata


def test_extra_options_are_forwarded_to_the_carver(green_image):
    # multi_start tries every start column (including single-start's), so its first
    # seam can never cost more -- this only holds if multi_start actually reached greedy.
    image = rand_img(9, 11, seed=0)
    single = carve_both_directions(get_algorithm("greedy"), image, 3)
    multi = carve_both_directions(get_algorithm("greedy"), image, 3, multi_start=True)
    assert multi.metadata["seam_energies"][0] < single.metadata["seam_energies"][0]


# ---------------------------------------------------------------- helpers


def test_seam_pixels_vertical_and_horizontal():
    rows, cols = _seam_pixels(np.array([2, 3, 4]), "vertical")
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [2, 3, 4]
    rows, cols = _seam_pixels(np.array([2, 3, 4]), "horizontal")
    assert rows.tolist() == [2, 3, 4] and cols.tolist() == [0, 1, 2]


def test_delete_vertical_and_horizontal():
    grid = np.arange(12).reshape(3, 4)
    v = _delete(grid, np.arange(3), np.array([0, 1, 3]), "vertical")
    assert v.tolist() == [[1, 2, 3], [4, 6, 7], [8, 9, 10]]
    h = _delete(grid, np.array([0, 1, 2, 2]), np.arange(4), "horizontal")
    assert h.tolist() == [[4, 1, 2, 3], [8, 9, 6, 7]]
