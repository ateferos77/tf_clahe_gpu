"""CLAHEConfig, input validation and environment introspection."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from gpu_clahe import CLAHEConfig, get_gpu_info, total_gpu_memory_mb, validate_input
from gpu_clahe.utils import require_valid_input


# --------------------------------------------------------------------------- #
# CLAHEConfig
# --------------------------------------------------------------------------- #
def test_defaults_are_valid() -> None:
    config = CLAHEConfig()
    assert config.tile_size == 32
    assert config.clip_limit == pytest.approx(0.035)
    assert config.dtype == tf.uint8


@pytest.mark.parametrize("tile_size", [0, 1, -4])
def test_invalid_tile_size_is_rejected(tile_size: int) -> None:
    with pytest.raises(ValueError, match="tile_size"):
        CLAHEConfig(tile_size=tile_size)


@pytest.mark.parametrize("clip_limit", [0.0, -0.1])
def test_invalid_clip_limit_is_rejected(clip_limit: float) -> None:
    with pytest.raises(ValueError, match="clip_limit"):
        CLAHEConfig(clip_limit=clip_limit)


def test_auto_batch_size_requires_a_batched_shape() -> None:
    with pytest.raises(ValueError, match="batched shape"):
        CLAHEConfig().auto_batch_size((512, 512))


def test_auto_batch_size_never_exceeds_the_dataset() -> None:
    assert CLAHEConfig().auto_batch_size((5, 64, 64)) <= 5


def test_auto_batch_size_falls_back_without_a_gpu() -> None:
    """With no readable device memory the result must still be usable."""
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=0):
        assert CLAHEConfig().auto_batch_size((1000, 512, 512)) == 32


def test_auto_batch_size_shrinks_as_images_grow() -> None:
    """Bigger images must not produce a bigger batch."""
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=8000):
        config = CLAHEConfig()
        small = config.auto_batch_size((10_000, 128, 128))
        large = config.auto_batch_size((10_000, 4096, 4096))
    assert small > large >= 1


@pytest.mark.parametrize(
    ("side", "expected"), [(256, 128), (512, 32), (1024, 8), (2048, 2)]
)
def test_auto_batch_size_caps_on_area_not_image_count(side: int, expected: int) -> None:
    """The cap tracks pixels per call, so it scales with the image size.

    Past ~8M pixels per call the GPU is already saturated and a larger batch
    only makes the host round trip worse. A flat image-count cap would be wrong
    at one end or the other: end-to-end throughput peaks near 128 images at
    256x256 but near 16-32 at 512x512.
    """
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=80_000):
        batch = CLAHEConfig().auto_batch_size((10_000, side, side))
    assert batch == expected


def test_auto_batch_size_respects_vram_over_the_area_cap() -> None:
    """On a small card the memory limit must win, not the area ceiling."""
    total_mb = 128
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=total_mb):
        batch = CLAHEConfig().auto_batch_size((10_000, 256, 256))
    working_set_mb = batch * 256 * 256 * 24 / (1024 * 1024)
    assert working_set_mb <= total_mb * 0.6
    assert batch < 128, "VRAM should bind here, not the 8M-pixel area cap"


def test_auto_batch_size_is_a_power_of_two() -> None:
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=8000):
        batch = CLAHEConfig().auto_batch_size((10_000, 512, 512))
    assert batch & (batch - 1) == 0, f"{batch} is not a power of two"


def test_auto_batch_size_stays_within_budget() -> None:
    """The chosen batch must actually fit the modelled working set."""
    total_mb = 4096
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=total_mb):
        batch = CLAHEConfig().auto_batch_size((10_000, 512, 512))
    working_set_mb = batch * 512 * 512 * 24 / (1024 * 1024)
    assert working_set_mb <= total_mb * 0.6


def test_total_gpu_memory_is_non_negative() -> None:
    assert total_gpu_memory_mb() >= 0


# --------------------------------------------------------------------------- #
# validate_input
# --------------------------------------------------------------------------- #
def test_accepts_well_formed_arrays(rng: np.random.Generator) -> None:
    valid, message = validate_input(
        rng.integers(0, 256, size=(2, 32, 32), dtype=np.uint8)
    )
    assert valid and message == ""


def test_accepts_well_formed_tensors(rng: np.random.Generator) -> None:
    images = tf.constant(rng.integers(0, 256, size=(2, 32, 32, 1), dtype=np.uint8))
    valid, _ = validate_input(images)
    assert valid


@pytest.mark.parametrize(
    ("images", "expected"),
    [
        (np.zeros((32, 32), np.uint8), "3-D"),
        (np.zeros((0, 32, 32), np.uint8), "Empty"),
        (np.zeros((2, 32, 32, 3), np.uint8), "single-channel"),
        (np.zeros((2, 32, 32), np.complex64), "Unsupported dtype"),
        ("not an array", "numpy array"),
    ],
)
def test_rejects_malformed_input(images, expected: str) -> None:
    valid, message = validate_input(images)
    assert not valid
    assert expected in message


def test_flags_zero_to_one_normalised_floats() -> None:
    """The classic silent failure: [0, 1] floats clamp to the bottom LUT bins.

    Nothing downstream raises, so validation is the only place this can be
    caught before it turns into an image of noise.
    """
    images = np.random.default_rng(0).random((2, 32, 32)).astype(np.float32)
    valid, message = validate_input(images)
    assert not valid
    assert "[0, 1]" in message and "255" in message


def test_accepts_floats_on_the_full_scale(rng: np.random.Generator) -> None:
    images = rng.uniform(0, 255, size=(2, 32, 32)).astype(np.float32)
    valid, _ = validate_input(images)
    assert valid


def test_require_valid_input_raises_with_the_same_message() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        require_valid_input(np.zeros((2, 32, 32, 3), np.uint8))


def test_require_valid_input_passes_good_data(rng: np.random.Generator) -> None:
    require_valid_input(rng.integers(0, 256, size=(2, 32, 32), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_gpu_info_reports_the_build() -> None:
    info = get_gpu_info()
    assert info["tensorflow_version"] == tf.__version__
    assert isinstance(info["num_gpus"], int)
    assert len(info["gpus"]) == info["num_gpus"]


@pytest.mark.gpu
def test_gpu_info_describes_the_device() -> None:
    info = get_gpu_info()
    assert info["num_gpus"] >= 1
    assert info["gpus"][0]["name"]
