"""User-facing batching driver around the :mod:`gpu_clahe.core` kernel."""

from __future__ import annotations

from typing import Union

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from .config import CLAHEConfig
from .core import clahe_gpu, setup_gpu
from .utils import require_valid_input

__all__ = ["ImageArray", "convert_clahe"]

ImageArray = Union[NDArray[np.generic], tf.Tensor]


def _as_batch_tensor(batch: ImageArray, dtype: tf.DType) -> tf.Tensor:
    """Convert a host batch to a tensor of ``dtype`` without dtype surprises.

    ``tf.convert_to_tensor`` refuses to change dtype, so an explicit cast is
    required; passing ``dtype=`` directly raises on any float input.
    """
    tensor = tf.convert_to_tensor(batch)
    return tensor if tensor.dtype == dtype else tf.cast(tensor, dtype)


def _pad_to(batch: tf.Tensor, size: int) -> tf.Tensor:
    """Right-pad the batch axis up to ``size`` by repeating the last image.

    Keeps every call to the kernel the same shape, so XLA compiles exactly once
    instead of recompiling for the short final batch. Repeating the last image
    (rather than zero-filling) keeps the padded slots cheap and numerically
    uninteresting; they are discarded immediately afterwards.

    The length is read from the *static* shape. Going through ``tf.shape``
    instead enqueues a device op and blocks on its result, which costs more per
    batch than the padding it is guarding.
    """
    current = batch.shape[0]
    if current is None:  # pragma: no cover - the driver always slices to size
        current = int(tf.shape(batch)[0])
    missing = size - int(current)
    if missing <= 0:
        return batch
    return tf.concat([batch, tf.repeat(batch[-1:], missing, axis=0)], axis=0)


def _run_batched(
    images: ImageArray,
    out: NDArray[np.generic],
    batch_size: int,
    tile_size: int,
    clip_limit: float,
    dtype: tf.DType,
) -> None:
    """Simple host-side slicing loop. Lowest latency for small workloads."""
    total = len(images)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch = _as_batch_tensor(images[start:stop], dtype)
        processed = clahe_gpu(
            _pad_to(batch, batch_size),
            tile_size=tile_size,
            clip_limit=clip_limit,
            dtype=dtype,
        )
        # The slice is a host-side view and only trims anything on the tail
        # batch, so there is nothing to branch on.
        out[start:stop] = processed.numpy()[: stop - start]


def _run_pipeline(
    images: ImageArray,
    out: NDArray[np.generic],
    batch_size: int,
    tile_size: int,
    clip_limit: float,
    dtype: tf.DType,
) -> None:
    """tf.data path: overlaps host-to-device copies with compute.

    ``drop_remainder=True`` keeps every batch the same static shape (so XLA
    compiles once); the leftover tail is handled by the batched path afterwards.

    Costs a second copy of the dataset: ``from_tensor_slices`` materialises the
    array it is handed, on top of the input and the output buffer. For anything
    close to filling host RAM, the plain batched path is the one that fits.
    """
    total = len(images)
    remainder = total % batch_size
    full = total - remainder

    if full:
        dataset = (
            tf.data.Dataset.from_tensor_slices(images[:full])
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        for index, batch in enumerate(dataset):
            processed = clahe_gpu(
                _as_batch_tensor(batch, dtype),
                tile_size=tile_size,
                clip_limit=clip_limit,
                dtype=dtype,
            )
            start = index * batch_size
            out[start : start + batch_size] = processed.numpy()

    if remainder:
        _run_batched(
            images[full:], out[full:], batch_size, tile_size, clip_limit, dtype
        )


def convert_clahe(
    images: ImageArray,
    batch_size: int | None = None,
    tile_size: int | None = None,
    clip_limit: float | None = None,
    use_pipeline: bool = False,
    return_tensor: bool = False,
    dtype: tf.DType | None = None,
    config: CLAHEConfig | None = None,
    validate: bool = True,
) -> ImageArray:
    """Apply CLAHE to a whole dataset, batching to fit in GPU memory.

    Args:
        images: ``(N, H, W)`` or ``(N, H, W, 1)`` array or tensor. Values are
            interpreted on a ``[0, 255]`` scale whatever the dtype.
        batch_size: Images per GPU call. ``None`` derives one from available VRAM
            and the image size via :meth:`CLAHEConfig.auto_batch_size`.
        tile_size: CLAHE tile side length in pixels. ``None`` takes it from
            ``config``, or 32.
        clip_limit: Contrast limit as a fraction of the pixels in a tile.
            ``None`` takes it from ``config``, or 0.035.
        use_pipeline: Route batches through ``tf.data``. Off by default: for an
            array already in host memory it measures slower than plain slicing,
            because ``from_tensor_slices`` copies the array and the prefetching
            has no I/O to hide. Worth enabling when the source is slow (decoding
            from disk), where overlapping input with compute does pay.
        return_tensor: Return a ``tf.Tensor`` instead of a NumPy array.
        dtype: Output dtype. ``None`` takes it from ``config``, or ``tf.uint8``.
        config: Supplies defaults for ``tile_size``, ``clip_limit``, ``dtype``,
            ``memory_growth`` and ``enable_xla``. Any argument passed
            explicitly still wins, including one that equals the default.
        validate: Run :func:`gpu_clahe.validate_input` first. On by default: the
            failure it catches - values that are not on a ``[0, 255]`` scale -
            is otherwise completely silent, since the kernel clamps and returns
            an array of the right shape and dtype containing nothing. Pass
            ``False`` only to skip the check knowingly.

    Returns:
        Processed images in the requested container, same shape as the input.

    Raises:
        ValueError: If ``images`` is empty, ``batch_size`` is not positive, or
            ``validate`` is set and the input fails validation.
    """
    # ``None`` means "not supplied", so an explicit argument wins even when it
    # happens to equal the default. Detecting that by comparing against the
    # default value instead would let the config quietly override an explicit
    # tile_size=32.
    defaults = config if config is not None else CLAHEConfig()
    if tile_size is None:
        tile_size = defaults.tile_size
    if clip_limit is None:
        clip_limit = defaults.clip_limit
    if dtype is None:
        dtype = defaults.dtype

    setup_gpu(
        memory_growth=defaults.memory_growth,
        enable_xla=defaults.enable_xla,
    )

    total = len(images)
    if total == 0:
        raise ValueError("Cannot process an empty image array.")

    # Materialise on the host once: both paths slice it repeatedly, and slicing
    # a device tensor per batch would round-trip through the GPU needlessly.
    # Done before validating so the check always runs against the array, whose
    # value-range test is the stricter of the two.
    if isinstance(images, tf.Tensor):
        images = images.numpy()

    if validate:
        require_valid_input(images)

    if batch_size is None:
        batch_size = defaults.auto_batch_size(tuple(images.shape), tile_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    batch_size = min(batch_size, total)

    out = np.empty(images.shape, dtype=dtype.as_numpy_dtype)
    runner = _run_pipeline if use_pipeline else _run_batched
    runner(images, out, batch_size, tile_size, clip_limit, dtype)

    return tf.convert_to_tensor(out) if return_tensor else out
