"""Numerical correctness of the CLAHE kernel."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import gpu_clahe
from gpu_clahe.core import _pad_symmetric, _tile_axis_weights

from .reference import clahe_reference, reference_luts

SHAPES = [
    pytest.param((2, 64, 64), id="square-aligned"),
    pytest.param((2, 64, 96), id="rect-aligned"),
    pytest.param((3, 70, 50), id="needs-padding"),
    pytest.param((1, 33, 31), id="smaller-than-two-tiles"),
    pytest.param((2, 32, 32), id="exactly-one-tile"),
]

PARAMS = [
    pytest.param(32, 0.035, id="defaults"),
    pytest.param(16, 0.01, id="small-tile-tight-clip"),
    pytest.param(8, 0.1, id="tiny-tile-loose-clip"),
    pytest.param(64, 0.02, id="large-tile"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(("tile_size", "clip_limit"), PARAMS)
def test_matches_numpy_reference(
    rng: np.random.Generator, shape, tile_size: int, clip_limit: float
) -> None:
    """The kernel is bit-exact against the golden NumPy implementation.

    Exact equality is achievable because the LUT normalisation is integer
    arithmetic; see gpu_clahe.core._build_luts. A regression here means either a
    genuine algorithm change or that float arithmetic has crept back in.
    """
    images = rng.integers(0, 256, size=shape, dtype=np.uint8)
    expected = clahe_reference(images, tile_size, clip_limit)
    actual = gpu_clahe.clahe_gpu(
        tf.constant(images), tile_size=tile_size, clip_limit=clip_limit
    ).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(("tile_size", "clip_limit"), PARAMS)
def test_gpu_matches_cpu_bit_exactly(
    uniform_noise: np.ndarray, tile_size: int, clip_limit: float
) -> None:
    """CPU and GPU must agree exactly, not merely closely.

    They did not before the LUT was moved to integer arithmetic: XLA:GPU lowers
    float division to an approximate reciprocal, which tipped quotients sitting
    exactly on ``x.5`` the other way.
    """
    with tf.device("/CPU:0"):
        on_cpu = gpu_clahe.clahe_gpu(
            tf.constant(uniform_noise), tile_size=tile_size, clip_limit=clip_limit
        ).numpy()
    with tf.device("/GPU:0"):
        on_gpu = gpu_clahe.clahe_gpu(
            tf.constant(uniform_noise), tile_size=tile_size, clip_limit=clip_limit
        ).numpy()
    np.testing.assert_array_equal(on_gpu, on_cpu)


def test_jit_and_nojit_agree(uniform_noise: np.ndarray) -> None:
    """The XLA and non-XLA entry points compute the same thing."""
    jit = gpu_clahe.clahe_gpu(tf.constant(uniform_noise)).numpy()
    nojit = gpu_clahe.clahe_gpu_nojit(tf.constant(uniform_noise), 32, 0.035, tf.uint8)
    np.testing.assert_array_equal(jit, nojit.numpy())


def test_dynamic_batch_uses_the_one_hot_histogram(uniform_noise: np.ndarray) -> None:
    """A dynamic batch falls back to the one-hot path and must still be exact.

    The fast scatter histogram needs ``num_segments`` at compile time, so an
    unknown batch size selects a different implementation entirely. That
    fallback is only reachable through a signature with a dynamic leading
    dimension, which is why it needs its own test rather than riding along on
    the eager ones.
    """
    signature = [tf.TensorSpec([None, *uniform_noise.shape[1:]], tf.uint8)]

    @tf.function(input_signature=signature, autograph=False)
    def run(images: tf.Tensor) -> tf.Tensor:
        return gpu_clahe.clahe_gpu_nojit(images, 32, 0.035, tf.uint8)

    actual = run(tf.constant(uniform_noise)).numpy()
    np.testing.assert_array_equal(actual, clahe_reference(uniform_noise))


def test_jit_kernel_accepts_a_dynamic_batch(uniform_noise: np.ndarray) -> None:
    """The XLA kernel itself must handle an unknown batch size.

    The other dynamic-batch test goes through ``clahe_gpu_nojit``, which is a
    different code path: without XLA nothing fuses the one-hot into the
    reduction, so the ``(B, T, P, 256)`` intermediate really is materialised.
    That is the configuration whose memory cost the one-hot docstring warns
    about, and it is not the one users get from ``clahe_gpu``. This pins the
    compiled path, which is the one the warning assumes.
    """
    signature = [tf.TensorSpec([None, *uniform_noise.shape[1:]], tf.uint8)]

    @tf.function(input_signature=signature, autograph=False)
    def run(images: tf.Tensor) -> tf.Tensor:
        return gpu_clahe.clahe_gpu(images)

    actual = run(tf.constant(uniform_noise)).numpy()
    np.testing.assert_array_equal(actual, clahe_reference(uniform_noise))


@pytest.mark.parametrize(
    ("height", "width"), [(1, 1), (3, 5), (8, 9), (31, 33), (5, 2), (2, 7), (64, 64)]
)
@pytest.mark.parametrize("tile_size", [8, 32, 64])
def test_padding_matches_numpy_symmetric(
    rng: np.random.Generator, height: int, width: int, tile_size: int
) -> None:
    """``_pad_symmetric`` must equal ``np.pad(mode="symmetric")`` exactly.

    ``tf.pad`` rejects a SYMMETRIC pad wider than the axis, so this applies the
    padding in chunks. That is only correct because symmetric padding repeats
    with period ``2 * dim`` - true, but not obvious, and asserted here directly
    rather than inferred from the end-to-end results.
    """
    n_y, n_x = -(-height // tile_size), -(-width // tile_size)
    pad_y, pad_x = n_y * tile_size - height, n_x * tile_size - width

    images = rng.integers(0, 256, size=(2, height, width)).astype(np.int32)
    padded = _pad_symmetric(tf.constant(images), pad_y, pad_x).numpy()
    expected = np.pad(images, ((0, 0), (0, pad_y), (0, pad_x)), mode="symmetric")

    np.testing.assert_array_equal(padded, expected)


def test_one_hot_and_scatter_histograms_agree(rng: np.random.Generator) -> None:
    """The two histogram formulations must be interchangeable, not just close."""
    from gpu_clahe.core import (
        _extract_tiles,
        _histograms_by_one_hot,
        _histograms_by_scatter,
    )

    images = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
    tiles = _extract_tiles(tf.cast(tf.constant(images), tf.int32), 32, 2, 2)

    scatter = _histograms_by_scatter(tiles, 3, 4).numpy()
    one_hot = _histograms_by_one_hot(tiles).numpy()
    np.testing.assert_array_equal(scatter, one_hot)
    # Every pixel must be counted exactly once.
    assert np.all(scatter.sum(axis=-1) == 32 * 32)


# --------------------------------------------------------------------------- #
# Border interpolation - regression tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tile_size", [8, 16, 32])
def test_corner_region_uses_edge_tile_alone(
    rng: np.random.Generator, tile_size: int
) -> None:
    """Pixels in the outer half-tile map through the edge tile's LUT only.

    Regression test for a border bug: deriving the upper tile index from the
    already-clamped lower index made the top-left half-tile a 50/50 blend of
    tiles 0 and 1, using a weight computed for the non-existent tile -1. The
    symptom was a visible seam along the top and left edges.
    """
    height = width = tile_size * 4
    images = rng.integers(0, 256, size=(2, height, width), dtype=np.uint8)

    result = gpu_clahe.clahe_gpu(tf.constant(images), tile_size=tile_size).numpy()
    luts = reference_luts(images, tile_size=tile_size)

    half = tile_size // 2
    corner_values = images[:, :half, :half]
    # Tile 0 is the top-left tile; np.take maps each pixel through its LUT.
    expected = np.stack([luts[b, 0][corner_values[b]] for b in range(images.shape[0])])
    np.testing.assert_array_equal(result[:, :half, :half], expected)


@pytest.mark.parametrize("tile_size", [8, 16, 32])
def test_far_corner_uses_last_tile_alone(
    rng: np.random.Generator, tile_size: int
) -> None:
    """The bottom-right half-tile likewise collapses onto the final tile."""
    tiles = 4
    height = width = tile_size * tiles
    images = rng.integers(0, 256, size=(2, height, width), dtype=np.uint8)

    result = gpu_clahe.clahe_gpu(tf.constant(images), tile_size=tile_size).numpy()
    luts = reference_luts(images, tile_size=tile_size)

    half = tile_size // 2
    last = tiles * tiles - 1
    corner_values = images[:, -half:, -half:]
    expected = np.stack(
        [luts[b, last][corner_values[b]] for b in range(images.shape[0])]
    )
    np.testing.assert_array_equal(result[:, -half:, -half:], expected)


@pytest.mark.parametrize("tile_size", [8, 32])
def test_axis_weights_collapse_at_both_edges(tile_size: int) -> None:
    """Directly assert the index/weight invariant the borders depend on."""
    n_tiles = 4
    length = tile_size * n_tiles
    lower, upper, _ = _tile_axis_weights(length, tile_size, n_tiles)
    lower, upper = lower.numpy(), upper.numpy()

    half = tile_size // 2
    # Outer half-tiles: both indices coincide, so the blend weight is moot.
    assert np.all(lower[:half] == 0) and np.all(upper[:half] == 0)
    assert np.all(lower[-half:] == n_tiles - 1)
    assert np.all(upper[-half:] == n_tiles - 1)
    # Interior: the two indices straddle the pixel.
    interior = slice(half, length - half)
    assert np.all(upper[interior] == lower[interior] + 1)
    assert np.all((lower >= 0) & (upper <= n_tiles - 1))


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("height", "width"), [(3, 5), (8, 9), (31, 33), (1, 1)])
def test_images_far_smaller_than_a_tile(
    rng: np.random.Generator, height: int, width: int
) -> None:
    """An image under half a tile across needs a pad wider than the axis.

    ``tf.pad`` rejects a SYMMETRIC pad that wide, so this used to abort inside
    the XLA MirrorPad kernel rather than raise. The padding is now applied in
    chunks; see gpu_clahe.core._pad_symmetric.
    """
    images = rng.integers(0, 256, size=(2, height, width), dtype=np.uint8)
    result = gpu_clahe.clahe_gpu(tf.constant(images), tile_size=64).numpy()
    assert result.shape == images.shape
    np.testing.assert_array_equal(result, clahe_reference(images, tile_size=64))


@pytest.mark.parametrize("value", [0, 1, 128, 254, 255])
def test_constant_image_stays_constant(value: int) -> None:
    """A flat input has no local contrast, so the output must stay flat.

    The output value is *not* the input value - equalising a degenerate
    histogram is not the identity - but it must be uniform, with no tile seams.
    """
    images = np.full((2, 64, 64), value, dtype=np.uint8)
    result = gpu_clahe.clahe_gpu(tf.constant(images)).numpy()
    assert result.min() == result.max()


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stays_in_range(rng: np.random.Generator, shape) -> None:
    images = rng.integers(0, 256, size=shape, dtype=np.uint8)
    result = gpu_clahe.clahe_gpu(tf.constant(images)).numpy()
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0 and int(result.max()) <= 255


def test_batch_is_processed_independently(rng: np.random.Generator) -> None:
    """Each image gets its own histograms; batching must not mix them."""
    images = rng.integers(0, 256, size=(4, 64, 64), dtype=np.uint8)
    batched = gpu_clahe.clahe_gpu(tf.constant(images)).numpy()
    individually = np.concatenate(
        [gpu_clahe.clahe_gpu(tf.constant(images[i : i + 1])).numpy() for i in range(4)]
    )
    np.testing.assert_array_equal(batched, individually)


def test_repeated_calls_are_deterministic(uniform_noise: np.ndarray) -> None:
    first = gpu_clahe.clahe_gpu(tf.constant(uniform_noise)).numpy()
    second = gpu_clahe.clahe_gpu(tf.constant(uniform_noise)).numpy()
    np.testing.assert_array_equal(first, second)


def test_luts_are_monotonic(rng: np.random.Generator) -> None:
    """A CDF is non-decreasing, so every LUT must be too.

    A non-monotonic LUT would invert local intensity ordering, which is a
    visible artefact rather than a subtle numerical one.
    """
    images = rng.integers(0, 256, size=(2, 64, 64), dtype=np.uint8)
    luts = reference_luts(images)
    assert np.all(np.diff(luts.astype(np.int32), axis=-1) >= 0)


def test_increases_contrast_of_low_contrast_input(low_contrast: np.ndarray) -> None:
    """The point of CLAHE: a narrow histogram should get stretched out."""
    result = gpu_clahe.clahe_gpu(tf.constant(low_contrast)).numpy()
    assert result.std() > low_contrast.std() * 2


def test_clip_limit_controls_contrast(low_contrast: np.ndarray) -> None:
    """A tighter clip limit must not amplify contrast more than a loose one."""
    tight = gpu_clahe.clahe_gpu(tf.constant(low_contrast), clip_limit=0.005).numpy()
    loose = gpu_clahe.clahe_gpu(tf.constant(low_contrast), clip_limit=0.5).numpy()
    assert tight.std() <= loose.std()


# --------------------------------------------------------------------------- #
# Cross-check against an independent implementation
# --------------------------------------------------------------------------- #
def test_agrees_with_opencv_in_distribution(low_contrast: np.ndarray) -> None:
    """Sanity-check that this really is CLAHE, not just a self-consistent LUT.

    The golden reference shares this package's algorithm, so it cannot catch a
    conceptual error. OpenCV is genuinely independent, but differs in LUT
    normalisation (it scales by ``255 / tile_pixels`` without subtracting the
    CDF floor) and in redistribution details, so the outputs are close rather
    than equal. Correlation is the honest thing to assert.
    """
    cv2 = pytest.importorskip("cv2")

    tile_size = 32
    height, width = low_contrast.shape[1:]
    grid = (-(-width // tile_size), -(-height // tile_size))
    # tile_pixels * clip_limit counts, expressed as OpenCV's multiple-of-mean.
    clahe = cv2.createCLAHE(clipLimit=0.035 * 256, tileGridSize=grid)

    ours = gpu_clahe.clahe_gpu(tf.constant(low_contrast)).numpy()
    theirs = np.stack([clahe.apply(image) for image in low_contrast])

    correlation = np.corrcoef(ours.ravel(), theirs.ravel())[0, 1]
    assert correlation > 0.95, f"correlation with OpenCV was only {correlation:.3f}"
    assert np.mean(np.abs(ours.astype(int) - theirs.astype(int))) < 12


def test_reference_and_opencv_both_lift_contrast(low_contrast: np.ndarray) -> None:
    """Both implementations should move the input the same direction."""
    cv2 = pytest.importorskip("cv2")
    clahe = cv2.createCLAHE(clipLimit=8.96, tileGridSize=(3, 2))
    theirs = np.stack([clahe.apply(image) for image in low_contrast])
    ours = clahe_reference(low_contrast)
    assert ours.std() > low_contrast.std()
    assert theirs.std() > low_contrast.std()
