"""Behaviour every SeamCarver must share, checked for all algorithms at once."""

import numpy as np
import pytest

from src.algorithms import get_algorithm
from tests.conftest import is_row_subsequence, rand_img

# (algorithm name, extra kwargs)
CARVERS = [
    pytest.param("dp", {}, id="dp"),
    pytest.param("greedy", {}, id="greedy"),
    pytest.param("greedy", {"multi_start": True}, id="greedy-multistart"),
    pytest.param("graphcut", {}, id="graphcut"),
    pytest.param("graphcut", {"energy": "forward", "method": "edmonds_karp"}, id="graphcut-forward-ek"),
]

H, W = 10, 12


@pytest.fixture
def image():
    return rand_img(H, W, seed=7)


def run(name, extra, image, n, **kwargs):
    return get_algorithm(name).run(image, num_seams=n, **extra, **kwargs)


@pytest.mark.parametrize("name, extra", CARVERS)
@pytest.mark.parametrize("n", [1, 4, W - 1])
def test_vertical_output_shape_and_pixels(name, extra, image, n):
    out = run(name, extra, image, n).image
    assert out.shape == (H, W - n, 3)
    assert out.dtype == np.uint8
    for r in range(H):  # each row is the original row with n pixels deleted, order kept
        assert is_row_subsequence(image[r], out[r])


@pytest.mark.parametrize("name, extra", CARVERS)
@pytest.mark.parametrize("n", [1, H - 1])
def test_horizontal_output_shape_and_pixels(name, extra, image, n):
    out = run(name, extra, image, n, direction="horizontal").image
    assert out.shape == (H - n, W, 3)
    for c in range(W):
        assert is_row_subsequence(image[:, c], out[:, c])


@pytest.mark.parametrize("name, extra", CARVERS)
def test_horizontal_equals_vertical_on_transpose(name, extra, image):
    horiz = run(name, extra, image, 3, direction="horizontal").image
    vert_t = run(name, extra, image.swapaxes(0, 1).copy(), 3).image
    np.testing.assert_array_equal(horiz, vert_t.swapaxes(0, 1))


@pytest.mark.parametrize("name, extra", CARVERS)
def test_zero_seams_is_identity(name, extra, image):
    result = run(name, extra, image, 0)
    np.testing.assert_array_equal(result.image, image)
    assert result.metadata["seams_removed"] == 0


@pytest.mark.parametrize("name, extra", CARVERS)
def test_flat_image(name, extra):
    flat = np.full((H, W, 3), 99, dtype=np.uint8)
    out = run(name, extra, flat, 5).image
    assert out.shape == (H, W - 5, 3)
    assert (out == 99).all()


@pytest.mark.parametrize("name, extra", CARVERS)
def test_single_row_and_single_column_images(name, extra):
    assert run(name, extra, rand_img(1, 6), 2).image.shape == (1, 4, 3)
    assert run(name, extra, rand_img(6, 1), 2, direction="horizontal").image.shape == (4, 1, 3)


@pytest.mark.parametrize("name, extra", CARVERS)
@pytest.mark.parametrize(
    "n, direction",
    [(-1, "vertical"), (W, "vertical"), (W + 5, "vertical"), (H, "horizontal"), (-3, "horizontal")],
)
def test_too_many_or_negative_seams_raise_valueerror(name, extra, image, n, direction):
    with pytest.raises(ValueError):
        run(name, extra, image, n, direction=direction)


@pytest.mark.parametrize("name, extra", CARVERS)
def test_input_not_mutated(name, extra, image):
    before = image.copy()
    run(name, extra, image, 3, show_all_seams=True)
    run(name, extra, image, 3, direction="horizontal")
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("name, extra", CARVERS)
def test_all_seams_overlay_marks_exactly_removed_pixels(name, extra):
    image = np.zeros((H, W, 3), dtype=np.uint8)  # no red pixels to start with
    image[:, :, 1] = rand_img(H, W, seed=3, channels=0)
    result = run(name, extra, image, 4, show_all_seams=True)
    red = np.all(result.metadata["all_seams_image"] == (0, 0, 255), axis=2)
    assert red.sum() == 4 * H
    assert (red.sum(axis=1) == 4).all()  # 4 pixels in every row


@pytest.mark.parametrize("name, extra", CARVERS)
def test_step_callback_receives_every_step(name, extra, image):
    steps = []
    run(name, extra, image, 3, step_callback=lambda info: steps.append(info["step"]))
    assert steps == [1, 2, 3]


@pytest.mark.parametrize("name, extra", [p for p in CARVERS if p.values[0] != "graphcut"])
def test_both_directions_shape(name, extra, image):
    assert run(name, extra, image, 3, direction="both").image.shape == (H - 3, W - 3, 3)


@pytest.mark.xfail(
    strict=True,
    reason="GAP: graphcut does not support direction='both', but gui.py always calls run(direction='both')",
)
def test_graphcut_supports_both_directions(image):
    assert get_algorithm("graphcut").run(image, num_seams=3, direction="both").image.shape == (H - 3, W - 3, 3)
