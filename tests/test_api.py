"""The convert_clahe driver: shapes, dtypes, batching and error handling."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import gpu_clahe
from gpu_clahe import CLAHEConfig

from .reference import clahe_reference


@pytest.mark.parametrize("batch_size", [1, 3, 4, 7, 16, 64])
def test_batch_size_does_not_change_the_result(
    rng: np.random.Generator, batch_size: int
) -> None:
    """Batching is an implementation detail and must not be observable.

    Sizes that do not divide the dataset exercise the tail-padding path, which
    pads the final batch up to full width and then discards the extra slots.
    """
    images = rng.integers(0, 256, size=(13, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images, batch_size=batch_size)
    np.testing.assert_array_equal(result, clahe_reference(images))


@pytest.mark.parametrize("use_pipeline", [False, True])
def test_pipeline_and_batched_paths_agree(
    rng: np.random.Generator, use_pipeline: bool
) -> None:
    """The tf.data path and the plain loop must produce identical output."""
    images = rng.integers(0, 256, size=(11, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images, batch_size=4, use_pipeline=use_pipeline)
    np.testing.assert_array_equal(result, clahe_reference(images))


def test_accepts_tensor_input_and_returns_array(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(tf.constant(images), batch_size=2)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, clahe_reference(images))


def test_return_tensor_gives_a_tensor(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images, batch_size=2, return_tensor=True)
    assert isinstance(result, tf.Tensor)
    np.testing.assert_array_equal(result.numpy(), clahe_reference(images))


def test_four_dimensional_input_round_trips(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(4, 48, 48, 1), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images, batch_size=2)
    assert result.shape == images.shape
    np.testing.assert_array_equal(result[..., 0], clahe_reference(images[..., 0]))


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64]
)
def test_accepts_every_supported_input_dtype(rng: np.random.Generator, dtype) -> None:
    """Input dtype must not change the answer; values are what matter.

    This is the regression test for the old driver, which forced inputs through
    ``tf.convert_to_tensor(..., dtype=tf.uint8)`` and raised on anything float.
    """
    values = rng.integers(0, 256, size=(4, 48, 48))
    images = values.astype(dtype)
    result = gpu_clahe.convert_clahe(images, batch_size=2)
    np.testing.assert_array_equal(result, clahe_reference(values.astype(np.uint8)))


@pytest.mark.parametrize("out_dtype", [tf.uint8, tf.float32, tf.int32])
def test_output_dtype_is_honoured(rng: np.random.Generator, out_dtype) -> None:
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images, batch_size=2, dtype=out_dtype)
    assert result.dtype == out_dtype.as_numpy_dtype


def test_float_output_is_not_rounded(rng: np.random.Generator) -> None:
    """Integer outputs round; float outputs keep the interpolated value."""
    images = rng.integers(0, 256, size=(2, 64, 64), dtype=np.uint8)
    result = gpu_clahe.clahe_gpu(tf.constant(images), dtype=tf.float32).numpy()
    assert np.any(result != np.round(result)), "float output looks rounded"


def test_auto_batch_size_is_used_when_omitted(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(6, 48, 48), dtype=np.uint8)
    result = gpu_clahe.convert_clahe(images)
    np.testing.assert_array_equal(result, clahe_reference(images))


def test_config_supplies_defaults(rng: np.random.Generator) -> None:
    """CLAHEConfig was previously accepted nowhere; it must now take effect."""
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    config = CLAHEConfig(tile_size=16, clip_limit=0.01)
    result = gpu_clahe.convert_clahe(images, batch_size=2, config=config)
    np.testing.assert_array_equal(
        result, clahe_reference(images, tile_size=16, clip_limit=0.01)
    )


def test_explicit_arguments_beat_config(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    config = CLAHEConfig(tile_size=16)
    result = gpu_clahe.convert_clahe(images, batch_size=2, tile_size=8, config=config)
    np.testing.assert_array_equal(result, clahe_reference(images, tile_size=8))


def test_explicit_argument_equal_to_the_default_beats_config(
    rng: np.random.Generator,
) -> None:
    """An explicit tile_size=32 must win over a config, not look "unset".

    Precedence used to be inferred by comparing the argument against the
    default, so passing the default value explicitly was indistinguishable from
    omitting it and the config silently took over.
    """
    images = rng.integers(0, 256, size=(4, 48, 48), dtype=np.uint8)
    config = CLAHEConfig(tile_size=16, clip_limit=0.01)
    result = gpu_clahe.convert_clahe(
        images,
        batch_size=2,
        tile_size=32,
        clip_limit=0.035,
        dtype=tf.uint8,
        config=config,
    )
    np.testing.assert_array_equal(
        result, clahe_reference(images, tile_size=32, clip_limit=0.035)
    )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        gpu_clahe.convert_clahe(np.zeros((0, 32, 32), dtype=np.uint8))


def test_zero_to_one_floats_are_rejected(rng: np.random.Generator) -> None:
    """The driver must not silently destroy a mis-scaled image.

    Values are read on a [0, 255] scale, so a [0, 1] normalised float image is
    clamped into the bottom LUT bin and comes back as a single flat value - a
    solid black frame of the right shape and dtype. validate_input has always
    detected this; until it was wired in here, nothing ever called it.
    """
    images = rng.random((4, 32, 32)).astype(np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gpu_clahe.convert_clahe(images)


def test_integers_above_255_are_rejected(rng: np.random.Generator) -> None:
    """12-bit input is clipped, not rescaled, and must not pass quietly."""
    images = rng.integers(0, 4096, size=(4, 32, 32)).astype(np.uint16)
    with pytest.raises(ValueError, match="Rescale"):
        gpu_clahe.convert_clahe(images)


def test_validation_can_be_turned_off(rng: np.random.Generator) -> None:
    """The check is a guard rail, not a wall: opting out stays possible."""
    images = rng.random((4, 32, 32)).astype(np.float32)
    result = gpu_clahe.convert_clahe(images, validate=False)
    assert result.shape == images.shape
    # Exactly the collapse the check exists to prevent.
    assert len(np.unique(result)) == 1


def test_validation_leaves_well_formed_input_alone(rng: np.random.Generator) -> None:
    """Validation must not change results for input that was always fine."""
    images = rng.integers(0, 256, size=(6, 48, 48), dtype=np.uint8)
    checked = gpu_clahe.convert_clahe(images, batch_size=2)
    unchecked = gpu_clahe.convert_clahe(images, batch_size=2, validate=False)
    np.testing.assert_array_equal(checked, unchecked)
    np.testing.assert_array_equal(checked, clahe_reference(images))


def test_tensor_input_is_validated_too(rng: np.random.Generator) -> None:
    """A tf.Tensor must not be a way around the check."""
    images = tf.constant(rng.random((4, 32, 32)).astype(np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gpu_clahe.convert_clahe(images)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_non_positive_batch_size_is_rejected(
    rng: np.random.Generator, batch_size: int
) -> None:
    images = rng.integers(0, 256, size=(4, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="batch_size"):
        gpu_clahe.convert_clahe(images, batch_size=batch_size)


def test_multichannel_input_is_rejected(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-channel"):
        gpu_clahe.clahe_gpu(tf.constant(images))


@pytest.mark.parametrize("rank", [2, 5])
def test_wrong_rank_is_rejected(rng: np.random.Generator, rank: int) -> None:
    shape = (32, 32) if rank == 2 else (2, 32, 32, 1, 1)
    images = rng.integers(0, 256, size=shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="rank"):
        gpu_clahe.clahe_gpu(tf.constant(images))


def test_tiny_tile_size_is_rejected(rng: np.random.Generator) -> None:
    images = rng.integers(0, 256, size=(2, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="tile_size"):
        gpu_clahe.clahe_gpu(tf.constant(images), tile_size=1)


def test_dynamic_spatial_shape_gives_an_actionable_error() -> None:
    """XLA cannot trace unknown H/W, so say so instead of failing obscurely."""

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.uint8)])
    def run(images: tf.Tensor) -> tf.Tensor:
        return gpu_clahe.clahe_gpu(images)

    with pytest.raises(ValueError, match="static"):
        run.get_concrete_function()
