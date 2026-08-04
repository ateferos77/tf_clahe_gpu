"""GPU-accelerated CLAHE kernels.

The algorithm is standard Zuiderveld CLAHE, expressed entirely as dense
TensorFlow ops so the whole thing compiles to a handful of XLA kernels:

1. **Tiling.** The image is padded up to a whole multiple of ``tile_size`` and
   reshaped (no gather, no ``extract_patches``) into non-overlapping tiles.
2. **Histogram.** A 256-bin histogram is built per tile, clipped at
   ``clip_limit * tile_pixels``, and the clipped mass is redistributed. The
   cumulative histogram is normalised into a 256-entry lookup table per tile.
3. **Interpolation.** Every pixel is mapped through the LUTs of the four
   surrounding tile centres and the results are bilinearly blended.

Conventions worth knowing
-------------------------
* ``clip_limit`` is a **fraction of the pixels in a tile**, not OpenCV's
  ``clipLimit``. ``0.035`` at ``tile_size=32`` clips any bin above 35 counts.
  Passing OpenCV-style values (e.g. ``3.0``) disables clipping entirely and
  degrades CLAHE to plain AHE.
* Sample values are assumed to lie in ``[0, 255]`` regardless of dtype. Float
  images normalised to ``[0, 1]`` must be scaled up first;
  :func:`gpu_clahe.utils.validate_input` flags that case.
* Only single-channel images are supported. A 4-D input must have a trailing
  channel dimension of exactly 1.
"""

from __future__ import annotations

import contextlib

import tensorflow as tf

__all__ = ["clahe_gpu", "clahe_gpu_nojit", "setup_gpu"]

_NUM_BINS = 256
_MAX_VALUE = 255.0


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def _static_hw(images: tf.Tensor) -> tuple[int, int]:
    """Return the static ``(height, width)`` of a 3-D tensor.

    The kernel needs ``H`` and ``W`` at trace time: the tile count, and so every
    ``reshape`` in the tiling step, derives from them. Leaving them dynamic
    would defeat XLA's shape inference, so we fail loudly instead of silently
    falling back to a slow path.
    """
    height, width = images.shape[1], images.shape[2]
    if height is None or width is None:
        raise ValueError(
            "clahe_gpu requires static spatial dimensions; got shape "
            f"{images.shape}. Batch size may be dynamic, but height and width "
            "must be known at trace time (e.g. call set_shape((None, H, W)), or "
            "use padded_batch/resize before entering a tf.data pipeline)."
        )
    return int(height), int(width)


# --------------------------------------------------------------------------- #
# Stage 1 - tiling
# --------------------------------------------------------------------------- #
def _pad_symmetric(images: tf.Tensor, pad_y: int, pad_x: int) -> tf.Tensor:
    """Symmetrically pad the bottom and right edges by ``pad_y`` / ``pad_x``.

    Reflecting the border beats zero-filling, which would invent a spike at 0 in
    the edge tiles' histograms and darken the image margins.

    ``tf.pad`` refuses a SYMMETRIC pad wider than the axis it is padding, and
    that limit is reachable: an image smaller than one tile needs
    ``pad = tile_size - dim``, which exceeds ``dim`` whenever the image is under
    half a tile across. Padding in chunks lifts the restriction and produces
    exactly what ``numpy.pad(..., mode="symmetric")`` would. Each chunk at least
    doubles the axis, so this runs in log(pad / dim) steps at trace time.
    """
    while pad_y > 0 or pad_x > 0:
        step_y = min(pad_y, int(images.shape[1]))
        step_x = min(pad_x, int(images.shape[2]))
        images = tf.pad(images, [[0, 0], [0, step_y], [0, step_x]], mode="SYMMETRIC")
        pad_y -= step_y
        pad_x -= step_x
    return images


def _extract_tiles(
    images: tf.Tensor, tile_size: int, n_tiles_y: int, n_tiles_x: int
) -> tf.Tensor:
    """Reshape ``(B, pH, pW)`` into ``(B, n_tiles, tile_size ** 2)``.

    Pure reshape + transpose, so this costs one contiguous copy rather than a
    gather. ``pH`` and ``pW`` must already be exact multiples of ``tile_size``.
    """
    grouped = tf.reshape(images, [-1, n_tiles_y, tile_size, n_tiles_x, tile_size])
    # (B, ty, ts, tx, ts) -> (B, ty, tx, ts, ts) so each tile is contiguous.
    grouped = tf.transpose(grouped, [0, 1, 3, 2, 4])
    return tf.reshape(grouped, [-1, n_tiles_y * n_tiles_x, tile_size * tile_size])


# --------------------------------------------------------------------------- #
# Stage 2 - per-tile histograms and LUTs
# --------------------------------------------------------------------------- #
def _histograms_by_scatter(tiles: tf.Tensor, batch: int, n_tiles: int) -> tf.Tensor:
    """256-bin histogram per tile via a single scatter-add. ``(B, T, 256)`` int32.

    This is the fast path, and by a wide margin: it touches one element per
    pixel, where the one-hot formulation touches 256. On a GTX 1650 at
    32x512x512 it runs in 1.1 ms against 13.3 ms - and since the histogram is
    ~85% of the kernel's total time, that difference sets the throughput of the
    whole package.

    Each tile is given its own 256-wide slice of a flat segment space, so one
    ``unsorted_segment_sum`` bins every tile of every image at once. Adding
    int32 ones makes the scatter exactly reproducible despite the atomics -
    integer addition is associative, so ordering cannot change the result. The
    same trick in float would not be deterministic.

    Requires a static batch size: XLA needs ``num_segments`` at compile time.
    """
    offsets = tf.range(batch * n_tiles, dtype=tf.int32) * _NUM_BINS
    segments = tf.reshape(tiles, [batch * n_tiles, -1]) + tf.expand_dims(offsets, 1)
    segments = tf.reshape(segments, [-1])

    counts = tf.math.unsorted_segment_sum(
        tf.ones_like(segments), segments, batch * n_tiles * _NUM_BINS
    )
    return tf.reshape(counts, [batch, n_tiles, _NUM_BINS])


def _histograms_by_one_hot(tiles: tf.Tensor) -> tf.Tensor:
    """256-bin histogram per tile, without needing a static batch size.

    ~12x slower than the scatter path, so this only runs when the batch size is
    dynamic. XLA fuses the one-hot into the reduction, so the ``(B, T, P, 256)``
    intermediate is never materialised - without that fusion this would need
    tens of GB.
    """
    return tf.reduce_sum(tf.one_hot(tiles, _NUM_BINS, dtype=tf.int32), axis=2)


def _histograms(tiles: tf.Tensor, batch: int | None, n_tiles: int) -> tf.Tensor:
    """Per-tile histograms, picking the fastest formulation the shapes allow."""
    if batch is None:
        return _histograms_by_one_hot(tiles)
    return _histograms_by_scatter(tiles, batch, n_tiles)


def _clip_and_redistribute(histograms: tf.Tensor, clip_value: int) -> tf.Tensor:
    """Clip each bin at ``clip_value`` and spread the excess over all bins.

    Single-pass redistribution: the excess is divided evenly across the 256 bins
    and the integer remainder is dropped. Re-clipping after redistribution (as
    OpenCV does not, but some implementations do) is deliberately skipped - it
    costs another pass and moves results by at most one count per bin.
    """
    excess = tf.reduce_sum(
        tf.maximum(histograms - clip_value, 0), axis=-1, keepdims=True
    )
    return tf.minimum(histograms, clip_value) + excess // _NUM_BINS


def _build_luts(histograms: tf.Tensor) -> tf.Tensor:
    """Turn clipped histograms into ``(B, n_tiles, 256)`` uint8 LUTs.

    Computes the textbook normalisation
    ``(cdf - cdf[0]) * 255 / (cdf[-1] - cdf[0])`` in **exact integer
    arithmetic**. The float spelling is tempting but not portable: XLA:GPU
    lowers float division to an approximate reciprocal, so a quotient that is
    exactly ``x.5`` on the CPU comes out a fraction below it on the GPU and
    rounds the other way. That made the same input produce different pixels on
    different devices. Integer division has no such freedom.

    ``(2 * 255 * n + s) // (2 * s)`` is round-half-up of ``255 * n / s``.

    Overflow: ``n`` is at most ``tile_size ** 2``, so the numerator stays below
    ``2**31`` for any ``tile_size`` up to ~2000 - far past anything useful.

    Note this differs from OpenCV, which scales by ``255 / tile_pixels`` without
    subtracting the floor, so outputs agree closely but not bit-exactly.
    """
    cdf = tf.cumsum(histograms, axis=-1)
    cdf_min = cdf[..., :1]
    numerator = cdf - cdf_min

    # A flat tile has cdf_max == cdf_min; clamping the span to 1 maps it to
    # all-zeros instead of dividing by zero.
    span = tf.maximum(cdf[..., -1:] - cdf_min, 1)

    lut = (2 * 255 * numerator + span) // (2 * span)
    # numerator <= span always, so the clip only guards against a malformed
    # histogram; the arithmetic above cannot exceed 255 on its own. uint8 is
    # therefore lossless here, and it quarters the memory traffic through the
    # per-pixel gathers in _apply_luts, which is the kernel's largest stage.
    return tf.cast(tf.clip_by_value(lut, 0, 255), tf.uint8)


# --------------------------------------------------------------------------- #
# Stage 3 - bilinear LUT interpolation
# --------------------------------------------------------------------------- #
def _tile_axis_weights(
    length: int, tile_size: int, n_tiles: int
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Per-pixel lower/upper tile index and blend weight along one axis.

    Pixel ``p`` sits at continuous tile coordinate ``p / tile_size - 0.5``, which
    puts tile ``k``'s centre exactly at coordinate ``k``.

    Both indices are clamped from the *unclamped* floor, and that detail is what
    makes the borders correct: in the outer half-tile the floor is ``-1``, so
    lower and upper both clamp to ``0`` and the blend collapses onto the edge
    tile's LUT whatever the weight is. Deriving the upper index from the
    already-clamped lower index instead blends tiles 0 and 1 using a weight
    computed for tiles -1 and 0, which smears the top and left edges.
    """
    coord = tf.range(length, dtype=tf.float32) / float(tile_size) - 0.5
    floor = tf.floor(coord)
    weight = coord - floor

    base = tf.cast(floor, tf.int32)
    lower = tf.clip_by_value(base, 0, n_tiles - 1)
    upper = tf.clip_by_value(base + 1, 0, n_tiles - 1)
    return lower, upper, weight


def _apply_luts(
    luts: tf.Tensor,
    values: tf.Tensor,
    height: int,
    width: int,
    tile_size: int,
    n_tiles_y: int,
    n_tiles_x: int,
) -> tf.Tensor:
    """Map every pixel through its four neighbouring LUTs and blend them.

    The LUTs are flattened to ``(B, n_tiles * 256)`` and addressed by the
    combined index ``tile * 256 + value``, so each pixel costs one scalar
    gather. The obvious alternative - gather each pixel's whole 256-entry LUT,
    then index it - materialises a ``(B, H, W, 256)`` tensor, i.e. 34 GB for a
    128x512x512 batch, which is what previously forced tiny batches.
    """
    iy0, iy1, wy = _tile_axis_weights(height, tile_size, n_tiles_y)
    ix0, ix1, wx = _tile_axis_weights(width, tile_size, n_tiles_x)

    # Tile indices are shared across the batch, so they stay (H, 1) / (1, W) and
    # broadcast rather than being materialised per image.
    row0 = tf.expand_dims(iy0 * n_tiles_x, axis=1)
    row1 = tf.expand_dims(iy1 * n_tiles_x, axis=1)
    col0 = tf.expand_dims(ix0, axis=0)
    col1 = tf.expand_dims(ix1, axis=0)

    luts_flat = tf.reshape(luts, [-1, n_tiles_y * n_tiles_x * _NUM_BINS])

    def lookup(tile_index: tf.Tensor) -> tf.Tensor:
        # (1, H, W) tile offsets broadcast against (B, H, W) values; XLA fuses
        # the arithmetic into the gather so no index tensor is ever stored.
        flat_index = tf.expand_dims(tile_index, 0) * _NUM_BINS + values
        return tf.cast(tf.gather(luts_flat, flat_index, batch_dims=1), tf.float32)

    wy = tf.expand_dims(wy, axis=1)  # (H, 1), broadcasts across width
    top = lookup(row0 + col0) * (1.0 - wx) + lookup(row0 + col1) * wx
    bottom = lookup(row1 + col0) * (1.0 - wx) + lookup(row1 + col1) * wx
    return top * (1.0 - wy) + bottom * wy


# --------------------------------------------------------------------------- #
# Public kernel
# --------------------------------------------------------------------------- #
def _clahe_impl(
    images: tf.Tensor, tile_size: int, clip_limit: float, dtype: tf.DType
) -> tf.Tensor:
    rank = images.shape.rank
    if rank is None:
        raise ValueError("clahe_gpu requires a tensor of known rank (3 or 4).")
    if rank not in (3, 4):
        raise ValueError(
            f"Expected a 3-D (B, H, W) or 4-D (B, H, W, 1) tensor, got rank {rank}."
        )

    expand_output = rank == 4
    if expand_output:
        channels = images.shape[-1]
        if channels is not None and channels != 1:
            raise ValueError(
                "Only single-channel images are supported; got a 4-D input with "
                f"{channels} channels. Convert to grayscale, or apply the kernel "
                "per channel (e.g. to the L channel of LAB)."
            )
        images = tf.squeeze(images, axis=-1)

    height, width = _static_hw(images)
    if tile_size < 2:
        raise ValueError(f"tile_size must be >= 2, got {tile_size}.")

    # Values index into the LUT from here on, so clamp once, up front.
    values = tf.cast(
        tf.clip_by_value(tf.cast(images, tf.float32), 0.0, _MAX_VALUE), tf.int32
    )

    n_tiles_y = -(-height // tile_size)  # ceil division
    n_tiles_x = -(-width // tile_size)
    pad_y = n_tiles_y * tile_size - height
    pad_x = n_tiles_x * tile_size - width

    padded = _pad_symmetric(values, pad_y, pad_x)

    tile_pixels = tile_size * tile_size
    tiles = _extract_tiles(padded, tile_size, n_tiles_y, n_tiles_x)

    batch = images.shape[0]
    histograms = _histograms(
        tiles, None if batch is None else int(batch), n_tiles_y * n_tiles_x
    )
    clip_value = max(1, int(tile_pixels * clip_limit))
    luts = _build_luts(_clip_and_redistribute(histograms, clip_value))

    # Interpolation reads the *unpadded* image: the padding existed only to make
    # the tile grid regular, so there is no reason to pay for it twice.
    result = _apply_luts(luts, values, height, width, tile_size, n_tiles_y, n_tiles_x)

    if dtype.is_integer:
        result = tf.round(result)
    result = tf.cast(tf.clip_by_value(result, 0.0, _MAX_VALUE), dtype)

    if expand_output:
        result = tf.expand_dims(result, axis=-1)
    return result


# reduce_retracing is deliberately off: it generalises the input signature, and
# once it has seen both a 3-D and a 4-D call it relaxes to unknown rank, which
# this kernel cannot trace. It would buy nothing anyway - XLA needs static
# shapes, so every distinct shape gets its own compilation either way.
#
# autograph is off because there is nothing for it to convert: every branch and
# loop here runs on Python values (ranks, tile counts, dtypes), never on tensor
# values. Leaving it on rewrites the source into a generated temp file, which
# buries real exceptions under generated frames and hides the module from
# coverage.
@tf.function(jit_compile=True, autograph=False)
def clahe_gpu(
    images: tf.Tensor,
    tile_size: int = 32,
    clip_limit: float = 0.035,
    dtype: tf.DType = tf.uint8,
) -> tf.Tensor:
    """Apply CLAHE to a batch of single-channel images.

    Args:
        images: ``(B, H, W)`` or ``(B, H, W, 1)`` tensor of any numeric dtype;
            values are interpreted on a ``[0, 255]`` scale. ``H`` and ``W`` must
            be static, ``B`` may be dynamic.
        tile_size: Side length in pixels of each contextual region. The image is
            symmetrically padded up to a multiple of this.
        clip_limit: Contrast limit as a **fraction of the pixels in a tile**
            (see module docstring). Values >= 1.0 disable clipping.
        dtype: Output dtype. Integer dtypes are rounded, floats are not.

    Returns:
        A tensor of ``dtype`` with the same shape as ``images``.

    Raises:
        ValueError: If rank, channel count, spatial shape or ``tile_size`` is
            unsupported.
    """
    return _clahe_impl(images, tile_size, clip_limit, dtype)


#: Non-XLA twin of :func:`clahe_gpu`. Useful when shapes cannot be made static,
#: and as the reference when diagnosing a suspected XLA miscompile.
clahe_gpu_nojit = tf.function(_clahe_impl, autograph=False)


def setup_gpu(memory_growth: bool = True, enable_xla: bool = True) -> bool:
    """Configure GPU memory growth and XLA; return True if a GPU is present.

    Memory growth can only be set before a GPU is initialised. Calling this
    after the first op has run leaves the existing setting in place, which is
    harmless and so is not treated as an error.
    """
    if enable_xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False

    if memory_growth:
        for gpu in gpus:
            # RuntimeError means the device is already initialised, in which
            # case the existing setting stands and there is nothing to do.
            with contextlib.suppress(RuntimeError):
                tf.config.experimental.set_memory_growth(gpu, True)
    return True
