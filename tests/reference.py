"""Straightforward NumPy CLAHE, used as the golden reference in tests.

This deliberately mirrors :mod:`gpu_clahe.core` step for step, in the most
obvious way possible: explicit loops for the histograms, no fusion, no tricks.
It is far too slow for real work, but it is easy to read and check by hand, so
when it and the TensorFlow kernel disagree the reference is the one to trust.

Keeping the two in lockstep is what lets the tests assert near-exact equality
instead of a vague tolerance. It does *not* independently validate that the
algorithm is CLAHE - ``test_correctness.py`` cross-checks against OpenCV for
that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUM_BINS = 256
MAX_VALUE = 255.0


def quantise(images: np.ndarray) -> np.ndarray:
    """Clamp to ``[0, 255]`` and convert to the int32 LUT indices."""
    return np.clip(images.astype(np.float32), 0.0, MAX_VALUE).astype(np.int32)


def tile_grid(height: int, width: int, tile_size: int) -> tuple[int, int]:
    """Number of tiles down and across."""
    return -(-height // tile_size), -(-width // tile_size)


def extract_tiles(values: np.ndarray, tile_size: int) -> np.ndarray:
    """Pad and split ``(B, H, W)`` into ``(B, n_tiles, tile_size ** 2)``."""
    batch, height, width = values.shape
    n_y, n_x = tile_grid(height, width, tile_size)
    padded = np.pad(
        values,
        ((0, 0), (0, n_y * tile_size - height), (0, n_x * tile_size - width)),
        mode="symmetric",
    )
    return (
        padded.reshape(batch, n_y, tile_size, n_x, tile_size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(batch, n_y * n_x, tile_size * tile_size)
    )


def reference_luts(
    images: np.ndarray, tile_size: int = 32, clip_limit: float = 0.035
) -> np.ndarray:
    """Per-tile lookup tables, ``(B, n_tiles, 256)`` int32.

    Exposed separately so tests can assert what a *specific* tile's LUT does to
    a pixel - which is how the border-interpolation behaviour is pinned down.
    """
    values = quantise(images)
    tiles = extract_tiles(values, tile_size)
    batch, n_tiles, _ = tiles.shape

    histograms = np.zeros((batch, n_tiles, NUM_BINS), dtype=np.int32)
    for b in range(batch):
        for t in range(n_tiles):
            histograms[b, t] = np.bincount(tiles[b, t], minlength=NUM_BINS)

    clip_value = max(1, int(tile_size * tile_size * clip_limit))
    excess = np.maximum(histograms - clip_value, 0).sum(axis=-1, keepdims=True)
    clipped = np.minimum(histograms, clip_value) + excess // NUM_BINS

    cdf = np.cumsum(clipped, axis=-1)
    cdf_min = cdf[..., :1]
    numerator = cdf - cdf_min
    span = np.maximum(cdf[..., -1:] - cdf_min, 1)

    # Integer round-half-up, matching the kernel exactly. See the note in
    # gpu_clahe.core._build_luts on why this is not done in floating point.
    lut: np.ndarray = np.clip((2 * 255 * numerator + span) // (2 * span), 0, 255)
    return lut.astype(np.int32)


def axis_weights(
    length: int, tile_size: int, n_tiles: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower index, upper index and blend weight for every pixel on one axis."""
    positions = np.arange(length, dtype=np.float32)
    coord = positions / np.float32(tile_size) - np.float32(0.5)
    floor = np.floor(coord)
    weight = coord - floor
    base = floor.astype(np.int32)
    return (
        np.clip(base, 0, n_tiles - 1),
        np.clip(base + 1, 0, n_tiles - 1),
        weight,
    )


def clahe_reference(
    images: np.ndarray,
    tile_size: int = 32,
    clip_limit: float = 0.035,
    dtype: npt.DTypeLike = np.uint8,
) -> np.ndarray:
    """CLAHE on a ``(B, H, W)`` array, matching :func:`gpu_clahe.clahe_gpu`."""
    values = quantise(images)
    batch, height, width = values.shape
    n_y, n_x = tile_grid(height, width, tile_size)

    luts = reference_luts(images, tile_size, clip_limit)
    flat = luts.reshape(batch, n_y * n_x * NUM_BINS)

    iy0, iy1, wy = axis_weights(height, tile_size, n_y)
    ix0, ix1, wx = axis_weights(width, tile_size, n_x)

    def lookup(tile_index: np.ndarray) -> np.ndarray:
        index = tile_index[None] * NUM_BINS + values
        taken = np.take_along_axis(flat, index.reshape(batch, -1), axis=1)
        return taken.reshape(batch, height, width).astype(np.float32)

    row0, row1 = (iy0 * n_x)[:, None], (iy1 * n_x)[:, None]
    col0, col1 = ix0[None, :], ix1[None, :]

    top = lookup(row0 + col0) * (1.0 - wx) + lookup(row0 + col1) * wx
    bottom = lookup(row1 + col0) * (1.0 - wx) + lookup(row1 + col1) * wx
    result = top * (1.0 - wy[:, None]) + bottom * wy[:, None]

    if np.issubdtype(dtype, np.integer):
        result = np.round(result)
    clipped: np.ndarray = np.clip(result, 0.0, MAX_VALUE)
    return clipped.astype(dtype)
