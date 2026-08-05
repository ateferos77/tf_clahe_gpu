"""Input validation and environment introspection."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

__all__ = ["get_gpu_info", "require_valid_input", "validate_input"]

_SUPPORTED_DTYPES = (
    np.uint8,
    np.uint16,
    np.int16,
    np.int32,
    np.float16,
    np.float32,
    np.float64,
)


_MAX_VALUE = 255.0


def _check_scale(peak: float, is_floating: bool) -> tuple[bool, str]:
    """Check that the sample values sit on the ``[0, 255]`` scale the kernel assumes.

    Both failures here are silent at every other layer: the kernel clamps to
    ``[0, 255]`` and carries on, so a mis-scaled image produces a plausible-
    looking tensor of the right shape and dtype containing nothing.
    """
    if is_floating and 0.0 < peak <= 1.0:
        return False, (
            f"Float input peaks at {peak:.3f}, which looks like a [0, 1] "
            "normalised image. Values are interpreted on a [0, 255] scale; "
            "multiply by 255 first."
        )
    if not is_floating and peak > _MAX_VALUE:
        return False, (
            f"Integer input peaks at {peak:.0f}, above the [0, 255] scale values "
            "are interpreted on. Everything above 255 would be clipped, not "
            "rescaled - a 12-bit image collapses almost entirely to one value. "
            "Rescale first, e.g. image / image.max() * 255."
        )
    return True, ""


def validate_input(images: np.ndarray | tf.Tensor) -> tuple[bool, str]:
    """Check that ``images`` is something the kernel can process.

    Args:
        images: Candidate input batch.

    Returns:
        ``(is_valid, message)``. ``message`` is empty when valid, and otherwise
        explains what to change.
    """
    if isinstance(images, tf.Tensor):
        shape = images.shape
        if shape.rank not in (3, 4):
            return False, f"Tensor must be 3-D or 4-D, got rank {shape.rank}."
        if shape.rank == 4 and shape[-1] not in (1, None):
            return False, (
                f"Only single-channel images are supported; got {shape[-1]} "
                "channels. Convert to grayscale first."
            )
        if shape[1] is None or shape[2] is None:
            return False, (
                f"Height and width must be static, got shape {shape}. "
                "Call set_shape((None, H, W)) before processing."
            )
        if images.dtype.as_numpy_dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(dt.__name__ for dt in _SUPPORTED_DTYPES)
            return False, (
                f"Unsupported dtype {images.dtype.name}; expected one of: {supported}."
            )
        if shape[0] == 0:
            return False, "Empty image array."
        # The scale check needs actual values. Inside a graph there are none, so
        # it is skipped rather than faked; eagerly, reduce_max stays on device
        # instead of copying the batch to the host just to look at one number.
        if tf.executing_eagerly() and shape[0] is not None:
            peak = float(tf.reduce_max(tf.cast(images, tf.float32)))
            return _check_scale(peak, images.dtype.is_floating)
        return True, ""

    if not isinstance(images, np.ndarray):
        return False, (
            f"Input must be a numpy array or tf.Tensor, got {type(images).__name__}."
        )

    if images.ndim not in (3, 4):
        return False, (
            "Images must be 3-D (batch, h, w) or 4-D (batch, h, w, 1), got "
            f"{images.ndim} dimensions."
        )
    if images.ndim == 4 and images.shape[-1] != 1:
        return False, (
            f"Only single-channel images are supported; got {images.shape[-1]} "
            "channels. Convert to grayscale first."
        )
    if images.shape[0] == 0:
        return False, "Empty image array."
    if images.dtype.type not in _SUPPORTED_DTYPES:
        supported = ", ".join(dt.__name__ for dt in _SUPPORTED_DTYPES)
        return False, f"Unsupported dtype {images.dtype}; expected one of: {supported}."

    return _check_scale(
        float(np.max(images)), bool(np.issubdtype(images.dtype, np.floating))
    )


def require_valid_input(images: np.ndarray | tf.Tensor) -> None:
    """Raise :class:`ValueError` if :func:`validate_input` rejects ``images``."""
    valid, message = validate_input(images)
    if not valid:
        raise ValueError(message)


def get_gpu_info() -> dict[str, Any]:
    """Summarise the visible GPUs and the TensorFlow build.

    Returns:
        A dict with the TF version, CUDA build flag, and per-GPU name and
        compute capability. Never raises; a probe failure lands in ``error``.
    """
    info: dict[str, Any] = {
        "tensorflow_version": tf.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "num_gpus": 0,
        "gpus": [],
    }

    try:
        gpus = tf.config.list_physical_devices("GPU")
        info["num_gpus"] = len(gpus)
        for gpu in gpus:
            # get_device_details is best-effort and returns {} on some builds.
            details = tf.config.experimental.get_device_details(gpu) or {}
            info["gpus"].append(
                {
                    "name": details.get("device_name", gpu.name),
                    "compute_capability": details.get("compute_capability"),
                }
            )
    except Exception as exc:  # pragma: no cover - depends on the local driver
        info["error"] = str(exc)

    return info
