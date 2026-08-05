"""CLAHEConfig, input validation and environment introspection."""

from __future__ import annotations

import subprocess
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from gpu_clahe import (
    CLAHEConfig,
    environment,
    get_gpu_info,
    gpu_driver_version,
    setup_gpu,
    total_gpu_memory_mb,
    validate_input,
)
from gpu_clahe.config import _bytes_per_pixel
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


@pytest.mark.parametrize(
    ("tile_size", "expected"), [(32, 27), (16, 36), (8, 72), (4, 216), (64, 24)]
)
def test_bytes_per_pixel_tracks_tile_size(tile_size: int, expected: int) -> None:
    """Histogram memory scales as 1/tile_size**2 and must be modelled.

    The buffers are (batch, n_tiles, 256) int32 and n_tiles is
    pixels / tile_size**2, so a single constant cannot describe them. Treating
    the cost as tile-independent understated the working set 9x at tile_size=4.
    """
    assert _bytes_per_pixel(tile_size) == expected


def test_auto_batch_size_shrinks_for_small_tiles() -> None:
    """A smaller tile means more histogram memory, so a smaller batch.

    Sized so VRAM binds rather than the 8M-pixel area cap; under the area cap
    memory is not the constraint and tile_size correctly makes no difference.
    """
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=512):
        big_tiles = CLAHEConfig(tile_size=32).auto_batch_size((10_000, 512, 512))
        small_tiles = CLAHEConfig(tile_size=8).auto_batch_size((10_000, 512, 512))

    assert small_tiles < big_tiles
    budget_mb = 512 * 0.6
    for tile_size, batch in ((32, big_tiles), (8, small_tiles)):
        working_set = batch * 512 * 512 * _bytes_per_pixel(tile_size) / (1024 * 1024)
        assert working_set <= budget_mb, f"tile_size={tile_size} overcommits"


def test_auto_batch_size_argument_overrides_the_config_field() -> None:
    """convert_clahe passes the tile_size it will actually use, not the field."""
    with mock.patch("gpu_clahe.config.total_gpu_memory_mb", return_value=512):
        config = CLAHEConfig(tile_size=32)
        assert config.auto_batch_size((10_000, 512, 512), tile_size=8) < (
            config.auto_batch_size((10_000, 512, 512))
        )


def test_total_gpu_memory_is_non_negative() -> None:
    assert total_gpu_memory_mb() >= 0


# --------------------------------------------------------------------------- #
# nvidia-smi probing - the real body never runs on a CI runner, so it is
# exercised here against a mocked subprocess rather than left unverified.
# --------------------------------------------------------------------------- #
@pytest.fixture
def pretend_gpu():
    """Make the nvidia-smi helpers believe a GPU and the binary are present."""
    with (
        mock.patch(
            "gpu_clahe.config.tf.config.list_physical_devices", return_value=["gpu"]
        ),
        mock.patch("gpu_clahe.config.shutil.which", return_value="/usr/bin/nvidia-smi"),
    ):
        yield


def _completed(stdout: str) -> mock.Mock:
    return mock.Mock(stdout=stdout)


@pytest.mark.usefixtures("pretend_gpu")
def test_total_gpu_memory_parses_nvidia_smi() -> None:
    with mock.patch(
        "gpu_clahe.config.subprocess.run", return_value=_completed("4096\n")
    ):
        assert total_gpu_memory_mb() == 4096


@pytest.mark.usefixtures("pretend_gpu")
def test_total_gpu_memory_takes_the_first_of_several_devices() -> None:
    with mock.patch(
        "gpu_clahe.config.subprocess.run", return_value=_completed("8192\n4096\n")
    ):
        assert total_gpu_memory_mb() == 8192


@pytest.mark.usefixtures("pretend_gpu")
@pytest.mark.parametrize("stdout", ["", "   \n", "not-a-number\n"])
def test_total_gpu_memory_survives_unusable_output(stdout: str) -> None:
    """Garbage must read as "unknown" (0), never as a memory figure."""
    with mock.patch("gpu_clahe.config.subprocess.run", return_value=_completed(stdout)):
        assert total_gpu_memory_mb() == 0


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 5),
        OSError("exec format error"),
    ],
)
@pytest.mark.usefixtures("pretend_gpu")
def test_total_gpu_memory_survives_a_failing_nvidia_smi(failure) -> None:
    with mock.patch("gpu_clahe.config.subprocess.run", side_effect=failure):
        assert total_gpu_memory_mb() == 0


def test_nvidia_smi_helpers_return_unknown_without_the_binary() -> None:
    with (
        mock.patch(
            "gpu_clahe.config.tf.config.list_physical_devices", return_value=["gpu"]
        ),
        mock.patch("gpu_clahe.config.shutil.which", return_value=None),
    ):
        assert total_gpu_memory_mb() == 0
        assert gpu_driver_version() is None


@pytest.mark.usefixtures("pretend_gpu")
def test_driver_version_is_reported() -> None:
    with mock.patch(
        "gpu_clahe.config.subprocess.run", return_value=_completed("535.309.01\n")
    ):
        assert gpu_driver_version() == "535.309.01"


def test_environment_records_what_a_result_needs() -> None:
    """A throughput figure is only checkable alongside all of these."""
    env = environment()
    for key in ("tensorflow_version", "python", "platform", "num_gpus"):
        assert key in env, f"environment() lost {key}"
    assert "driver_version" in env


# --------------------------------------------------------------------------- #
# setup_gpu
# --------------------------------------------------------------------------- #
def test_setup_gpu_reports_whether_a_gpu_exists(has_gpu: bool) -> None:
    assert setup_gpu() is has_gpu


def test_setup_gpu_can_leave_xla_alone() -> None:
    """enable_xla=False must not touch the global JIT setting."""
    with mock.patch("gpu_clahe.core.tf.config.optimizer.set_jit") as set_jit:
        setup_gpu(enable_xla=False)
        set_jit.assert_not_called()

        setup_gpu(enable_xla=True)
        set_jit.assert_called_once_with(True)


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


def test_flags_integers_above_the_255_scale() -> None:
    """A 12-bit image is clipped, not rescaled, and loses almost everything.

    Values are read on a [0, 255] scale whatever the dtype, so a 0-4095
    radiograph collapses ~94% of its pixels onto 255. Nothing downstream
    raises, which makes this the integer twin of the [0, 1] float trap.
    """
    images = (
        np.random.default_rng(0).integers(0, 4096, size=(2, 32, 32)).astype(np.uint16)
    )
    valid, message = validate_input(images)
    assert not valid
    assert "255" in message and "Rescale" in message


def test_accepts_integers_on_the_full_scale(rng: np.random.Generator) -> None:
    assert validate_input(rng.integers(0, 256, size=(2, 32, 32), dtype=np.uint8))[0]


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (
            lambda: np.random.default_rng(0).random((2, 32, 32)).astype(np.float32),
            "[0, 1]",
        ),
        (
            lambda: (
                np.random.default_rng(0)
                .integers(0, 4096, size=(2, 32, 32))
                .astype(np.uint16)
            ),
            "255",
        ),
        (lambda: np.zeros((2, 32, 32, 3), np.uint8), "single-channel"),
        (lambda: np.zeros((0, 32, 32), np.uint8), "Empty"),
        (lambda: np.zeros((2, 32, 32), np.complex64), "Unsupported dtype"),
    ],
)
def test_tensor_and_array_branches_agree(make, expected: str) -> None:
    """The same data must not be valid as a tensor and invalid as an array.

    The tensor branch used to check only rank, channels and static shape, so
    every value-level trap slipped through whenever the caller happened to
    hand over a tf.Tensor.
    """
    images = make()
    array_valid, array_message = validate_input(images)
    tensor_valid, tensor_message = validate_input(tf.constant(images))

    assert not array_valid and not tensor_valid
    assert expected in array_message
    assert expected in tensor_message


def test_tensor_branch_rejects_dynamic_spatial_dims() -> None:
    """H and W must be static; say so rather than failing later inside XLA."""

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.uint8)])
    def check(images: tf.Tensor) -> tf.Tensor:
        valid, message = validate_input(images)
        assert not valid
        assert "static" in message
        return images

    check.get_concrete_function()


def test_tensor_branch_skips_the_scale_check_in_a_graph() -> None:
    """Inside a graph there are no values to inspect, so shape checks only."""

    @tf.function(input_signature=[tf.TensorSpec([4, 32, 32], tf.float32)])
    def check(images: tf.Tensor) -> tf.Tensor:
        assert validate_input(images) == (True, "")
        return images

    check.get_concrete_function()


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
