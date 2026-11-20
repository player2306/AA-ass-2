"""Core behaviour: shared helpers, DP optimality, every algorithm's contract, I/O."""

import numpy as np
import pytest

from src.algorithms import carve_to_size, get_algorithm
from src.algorithms.dp import _compute_energy, _find_vertical_seam, _update_energy
from src.image_io import load_image, save_image
from src.utils import remove_vertical_seam, sobel_edge_map
from tests.conftest import best_seam_cost, is_subsequence, rand_img

ALGORITHMS = ["dp", "greedy", "graphcut"]
H, W = 10, 12


# ---------------------------------------------------------------- shared helpers


def test_remove_vertical_seam():
    image = rand_img(4, 5)
    seam = np.array([0, 1, 4, 4])
    out = remove_vertical_seam(image, seam)
    assert out.shape == (4, 4, 3)
    for r, c in enumerate(seam):
        np.testing.assert_array_equal(out[r], np.delete(image[r], c, axis=0))
    with pytest.raises(ValueError):
        remove_vertical_seam(image, np.array([0, 0, 5, 0]))  # out of bounds
    with pytest.raises(ValueError):
        remove_vertical_seam(image, np.zeros(3, dtype=int))  # wrong length


def test_energy_is_zero_on_flat_image():
    assert not sobel_edge_map(np.full((5, 6, 3), 123, np.uint8)).any()


# ---------------------------------------------------------------- DP correctness


def test_dp_seam_is_optimal_vs_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(25):
        energy = rng.integers(0, 10, size=(5, 4)).astype(np.float32)
        seam = _find_vertical_seam(energy)
        assert np.all(np.abs(np.diff(seam)) <= 1)  # connected
        assert energy[np.arange(5), seam].sum() == best_seam_cost(energy)


def test_dp_seam_does_not_wrap_around_edges():
    energy = np.full((2, 5), 50.0, dtype=np.float32)
    energy[0, 0] = energy[1, 4] = 0  # only "connected" through wrap-around
    seam = _find_vertical_seam(energy)
    assert energy[np.arange(2), seam].sum() == 50


def test_dp_incremental_energy_matches_full_recompute():
    image = rand_img(12, 15, seed=1)
    energy = _compute_energy(image)
    for _ in range(10):
        seam = _find_vertical_seam(energy)
        image = remove_vertical_seam(image, seam)
        keep = np.ones(energy.shape, bool)
        keep[np.arange(12), seam] = False
        energy = _update_energy(image, energy[keep].reshape(12, -1), seam)
        np.testing.assert_allclose(energy, _compute_energy(image), atol=1e-3)


# ---------------------------------------------------------------- every algorithm


@pytest.mark.parametrize("name", ALGORITHMS)
def test_carving_keeps_original_pixels_in_order(name):
    image = rand_img(H, W, seed=7)
    carver = get_algorithm(name)
    vertical = carver.run(image, num_seams=4).image
    assert vertical.shape == (H, W - 4, 3)
    assert all(is_subsequence(image[r], vertical[r]) for r in range(H))
    horizontal = carver.run(image, num_seams=3, direction="horizontal").image
    assert horizontal.shape == (H - 3, W, 3)
    assert all(is_subsequence(image[:, c], horizontal[:, c]) for c in range(W))
    assert carver.run(image, num_seams=2, direction="both").image.shape == (H - 2, W - 2, 3)


@pytest.mark.parametrize("name", ALGORITHMS)
def test_invalid_arguments_raise(name):
    carver, image = get_algorithm(name), rand_img(H, W)
    for kwargs in ({"num_seams": -1}, {"num_seams": W}, {"num_seams": H, "direction": "horizontal"},
                   {"num_seams": 1, "direction": "diagonal"}):
        with pytest.raises(ValueError):
            carver.run(image, **kwargs)


@pytest.mark.parametrize("name", ALGORITHMS)
def test_edge_inputs(name):
    carver = get_algorithm(name)
    image = rand_img(H, W, seed=3)
    before = image.copy()
    frames = []
    keep_frame = lambda info: frames.append(info["frame"])  # noqa: E731

    np.testing.assert_array_equal(carver.run(image, num_seams=0).image, image)  # 0 seams = unchanged
    # carve down to a single column / row, with step frames and overlay drawn
    assert carver.run(image, num_seams=W - 1, step_callback=keep_frame).image.shape == (H, 1, 3)
    assert carver.run(image, num_seams=H - 1, direction="horizontal", show_all_seams=True,
                      step_callback=keep_frame).image.shape == (1, W, 3)
    # a 1-column image carved horizontally (what carving to 1x1 does second)
    assert carver.run(image[:, :1].copy(), num_seams=2, direction="horizontal",
                      step_callback=keep_frame).image.shape == (H - 2, 1, 3)
    # grayscale input (2-D array)
    gray = carver.run(image[:, :, 0].copy(), num_seams=2, show_all_seams=True, step_callback=keep_frame)
    assert gray.image.shape == (H, W - 2)
    assert len(frames) == (W - 1) + (H - 1) + 2 + 2
    np.testing.assert_array_equal(image, before)  # input never modified


# ---------------------------------------------------------------- both directions / target size


def test_both_directions_overlay_marks_every_removed_pixel():
    image = np.zeros((9, 11, 3), np.uint8)
    image[:, :, 1] = rand_img(9, 11, channels=0)  # no pure-red pixels to start with
    result = get_algorithm("dp").run(image, num_seams=3, direction="both", show_all_seams=True)
    red = np.all(result.metadata["all_seams_image"] == (0, 0, 255), axis=2)
    assert red.sum() == 9 * 11 - 6 * 8


def test_carve_to_size():
    image, dp = rand_img(9, 11), get_algorithm("dp")
    assert carve_to_size(dp, image, 7, 5).image.shape == (5, 7, 3)
    assert carve_to_size(dp, image, 1, 1).image.shape == (1, 1, 3)
    for target in [(12, 9), (11, 10), (0, 5)]:
        with pytest.raises(ValueError, match="only shrink"):
            carve_to_size(dp, image, *target)


# ---------------------------------------------------------------- image I/O


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError):
        load_image(fake)


def test_save_image_roundtrip_and_bad_extension(tmp_path):
    image = rand_img(7, 9)
    np.testing.assert_array_equal(load_image(save_image(image, tmp_path / "a" / "b.png")), image)
    with pytest.raises(IOError):
        save_image(image, tmp_path / "out.notanimage")
