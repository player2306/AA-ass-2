"""Unit tests for the DP carver (src/algorithms/dp.py)."""

import numpy as np
import pytest

from src.algorithms.dp import DPSeamCarver, _compute_energy, _find_vertical_seam, _update_energy
from src.utils import remove_vertical_seam
from tests.conftest import assert_valid_vertical_seam, best_seam_cost, rand_img


def cost(energy, seam):
    return float(energy[np.arange(len(seam)), seam].sum())


# ---------------------------------------------------------------- _find_vertical_seam


@pytest.mark.parametrize("seed", range(25))
def test_seam_is_optimal_vs_brute_force(seed):
    energy = np.random.default_rng(seed).integers(0, 10, size=(5, 4)).astype(np.float32)
    seam = _find_vertical_seam(energy)
    assert_valid_vertical_seam(seam, 5, 4)
    assert cost(energy, seam) == pytest.approx(best_seam_cost(energy))


@pytest.mark.parametrize("col", [0, 3, 6], ids=["left-border", "middle", "right-border"])
def test_seam_follows_zero_energy_column(col):
    energy = np.full((6, 7), 9.0, dtype=np.float32)
    energy[:, col] = 0
    assert _find_vertical_seam(energy).tolist() == [col] * 6


def test_seam_follows_diagonal_path():
    energy = np.full((5, 5), 9.0, dtype=np.float32)
    energy[np.arange(5), np.arange(5)] = 0
    assert _find_vertical_seam(energy).tolist() == [0, 1, 2, 3, 4]


def test_seam_does_not_wrap_around_edges():
    # cheap pixels at opposite borders would only connect via wrap-around (np.roll bug)
    energy = np.full((2, 5), 50.0, dtype=np.float32)
    energy[0, 0] = 0
    energy[1, 4] = 0
    seam = _find_vertical_seam(energy)
    assert_valid_vertical_seam(seam, 2, 5)
    assert cost(energy, seam) == best_seam_cost(energy) == 50


def test_width_one_energy():
    assert _find_vertical_seam(np.ones((4, 1), np.float32)).tolist() == [0, 0, 0, 0]


def test_height_one_energy():
    assert _find_vertical_seam(np.array([[4, 2, 1, 3]], np.float32)).tolist() == [2]


def test_all_equal_energy_gives_valid_seam():
    assert_valid_vertical_seam(_find_vertical_seam(np.zeros((6, 6), np.float32)), 6, 6)


# ---------------------------------------------------------------- _update_energy


@pytest.mark.parametrize("seed", range(5))
def test_incremental_energy_update_matches_full_recompute(seed):
    image = rand_img(12, 15, seed=seed)
    energy = _compute_energy(image)
    h = image.shape[0]
    for _ in range(10):
        seam = _find_vertical_seam(energy)
        image = remove_vertical_seam(image, seam)
        keep = np.ones(energy.shape, bool)
        keep[np.arange(h), seam] = False
        energy = _update_energy(image, energy[keep].reshape(h, -1), seam)
        np.testing.assert_allclose(energy, _compute_energy(image), atol=1e-3)


# ---------------------------------------------------------------- DPSeamCarver.run


def test_zero_seams_returns_equal_copy(img):
    result = DPSeamCarver().run(img, num_seams=0)
    np.testing.assert_array_equal(result.image, img)
    assert result.image is not img
    assert result.metadata["seams"] == []


@pytest.mark.parametrize(
    "n, direction",
    [(-1, "vertical"), (12, "vertical"), (100, "vertical"), (10, "horizontal"), (-1, "horizontal")],
)
def test_invalid_seam_count_raises(img, n, direction):  # img is 10 high, 12 wide
    with pytest.raises(ValueError):
        DPSeamCarver().run(img, num_seams=n, direction=direction)


@pytest.mark.parametrize("direction", ["diagonal", "", "VERTICAL"])
def test_invalid_direction_raises(img, direction):
    with pytest.raises(ValueError, match="direction"):
        DPSeamCarver().run(img, num_seams=1, direction=direction)


def test_remove_all_but_one_column(img):
    assert DPSeamCarver().run(img, num_seams=11).image.shape == (10, 1, 3)


def test_remove_all_but_one_row(img):
    assert DPSeamCarver().run(img, num_seams=9, direction="horizontal").image.shape == (1, 12, 3)


def test_single_row_image():
    assert DPSeamCarver().run(rand_img(1, 6), num_seams=3).image.shape == (1, 3, 3)


def test_input_image_is_not_mutated(img):
    before = img.copy()
    DPSeamCarver().run(img, num_seams=4, show_all_seams=True)
    np.testing.assert_array_equal(img, before)


def test_metadata_seams_map_back_to_original_columns(img):
    n = 6
    result = DPSeamCarver().run(img, num_seams=n)
    meta = result.metadata
    assert meta["seams_removed"] == n
    assert meta["direction"] == "vertical"
    assert len(meta["seams"]) == len(meta["seams_original"]) == n
    stacked = np.stack(meta["seams_original"])  # (n, h)
    assert stacked.min() >= 0 and stacked.max() < img.shape[1]
    for r in range(img.shape[0]):
        removed = stacked[:, r]
        assert len(set(removed.tolist())) == n  # never removes the same pixel twice
        kept = np.delete(img[r], removed, axis=0)
        np.testing.assert_array_equal(result.image[r], kept)


def test_step_frames_only_kept_when_requested(img):
    assert DPSeamCarver().run(img, num_seams=3).metadata["step_frames"] == []
    frames = DPSeamCarver().run(img, num_seams=3, keep_steps=True).metadata["step_frames"]
    assert len(frames) == 3


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
def test_step_callback_called_once_per_seam(img, direction):
    calls = []
    DPSeamCarver().run(img, num_seams=4, direction=direction, step_callback=calls.append)
    assert [c["step"] for c in calls] == [1, 2, 3, 4]
    assert all(c["total"] == 4 for c in calls)
    # image passed to the callback is in normal orientation and shrinks each step
    if direction == "vertical":
        assert [c["image"].shape[:2] for c in calls] == [(10, 12), (10, 11), (10, 10), (10, 9)]
    else:
        assert [c["image"].shape[:2] for c in calls] == [(10, 12), (9, 12), (8, 12), (7, 12)]


@pytest.mark.parametrize("direction", ["vertical", "horizontal"])
def test_all_seams_overlay(img, direction):
    with_overlay = DPSeamCarver().run(img, num_seams=3, direction=direction, show_all_seams=True)
    assert with_overlay.metadata["all_seams_image"].shape == img.shape
    without = DPSeamCarver().run(img, num_seams=3, direction=direction)
    assert without.metadata["all_seams_image"] is None


def test_runtime_is_recorded(img):
    result = DPSeamCarver().run(img, num_seams=2)
    assert result.algorithm == "dp"
    assert result.runtime_s >= 0


@pytest.mark.parametrize("direction, expected", [("vertical", (8, 6)), ("horizontal", (5, 9))])
def test_grayscale_image(direction, expected):
    gray = rand_img(8, 9, channels=0)
    result = DPSeamCarver().run(gray, num_seams=3, direction=direction)
    assert result.image.shape == expected


def test_grayscale_incremental_energy_matches_full_recompute():
    image = rand_img(10, 12, seed=3, channels=0)
    energy = _compute_energy(image)
    seam = _find_vertical_seam(energy)
    image = remove_vertical_seam(image, seam)
    keep = np.ones(energy.shape, bool)
    keep[np.arange(10), seam] = False
    energy = _update_energy(image, energy[keep].reshape(10, -1), seam)
    np.testing.assert_allclose(energy, _compute_energy(image), atol=1e-3)


def test_grayscale_result_matches_color_result_on_gray_image():
    gray = rand_img(8, 9, seed=4, channels=0)
    color = np.dstack([gray, gray, gray])
    g = DPSeamCarver().run(gray, num_seams=3).image
    c = DPSeamCarver().run(color, num_seams=3).image
    np.testing.assert_array_equal(g, c[:, :, 0])
