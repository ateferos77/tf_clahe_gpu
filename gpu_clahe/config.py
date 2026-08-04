"""Configuration and automatic batch-size selection."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

import tensorflow as tf

__all__ = ["CLAHEConfig", "total_gpu_memory_mb"]

# Peak device memory the kernel touches per image, as a multiple of the image's
# pixel count. The kernel holds the int32 values, the padded copy, the
# tile-major copy and the float32 accumulator concurrently; XLA fuses the four
# gathers so they do not each add a buffer. Four live int32-sized buffers, then
# rounded up to six for headroom against allocator fragmentation.
_BYTES_PER_PIXEL_WORKING_SET = 4 * 6

# Never hand the whole card to one batch: the CUDA context, cuDNN workspaces and
# any co-resident model need room too.
_USABLE_MEMORY_FRACTION = 0.6

_FALLBACK_BATCH_SIZE = 32

# Past roughly this much work per call the batch axis stops buying anything: the
# GPU is already saturated, so a larger batch adds under 1% of kernel throughput
# (at 512x512, 13,862 -> 13,987 img/s going from 32 to 256 images) while making
# the host round trip markedly worse and leaving less memory for anything else
# on the card.
#
# The ceiling is expressed in *pixels*, not images, because that is what the
# measurements track: end-to-end throughput peaks near 128 images at 256x256 and
# near 16-32 at 512x512, which is roughly constant area. Capping on image count
# alone gives up 15-20% at one size or the other.
_MAX_AUTO_BATCH_PIXELS = 8 << 20


def total_gpu_memory_mb() -> int:
    """Total memory of the first visible GPU in MB, or 0 if unavailable.

    TensorFlow exposes only *current* and *peak* usage
    (``tf.config.experimental.get_memory_info``), never the device total, so
    this shells out to ``nvidia-smi``. A missing or uncooperative ``nvidia-smi``
    yields 0, which callers read as "unknown" rather than "no memory".
    """
    if not tf.config.list_physical_devices("GPU"):
        return 0

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return 0

    try:
        output = subprocess.run(
            [binary, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0

    lines = output.strip().splitlines()
    if not lines:
        return 0
    try:
        return int(lines[0].strip())
    except ValueError:
        return 0


@dataclass
class CLAHEConfig:
    """Processing parameters, with VRAM-aware batch sizing.

    Attributes:
        tile_size: CLAHE tile side length in pixels.
        clip_limit: Contrast limit as a fraction of the pixels in a tile.
        dtype: Output dtype.
        enable_xla: Enable XLA JIT globally from :func:`gpu_clahe.setup_gpu`.
        memory_growth: Grow GPU memory on demand rather than pre-allocating.
    """

    tile_size: int = 32
    clip_limit: float = 0.035
    dtype: tf.DType = tf.uint8
    enable_xla: bool = True
    memory_growth: bool = True

    def __post_init__(self) -> None:
        """Reject parameters the kernel cannot honour, at construction time."""
        if self.tile_size < 2:
            raise ValueError(f"tile_size must be >= 2, got {self.tile_size}.")
        if self.clip_limit <= 0:
            raise ValueError(f"clip_limit must be > 0, got {self.clip_limit}.")

    def auto_batch_size(self, image_shape: Sequence[int]) -> int:
        """Pick a batch size whose working set fits in VRAM.

        Args:
            image_shape: Shape of the whole dataset, ``(N, H, W)`` or
                ``(N, H, W, C)``.

        Returns:
            A batch size in ``[1, N]``, falling back to a conservative default
            when device memory cannot be determined.

        Raises:
            ValueError: If ``image_shape`` is not a batched image shape.
        """
        if len(image_shape) < 3:
            raise ValueError(
                f"Expected a batched shape (N, H, W[, C]), got {tuple(image_shape)}."
            )

        count = int(image_shape[0])
        total_mb = total_gpu_memory_mb()
        if total_mb <= 0:
            return max(1, min(_FALLBACK_BATCH_SIZE, count))

        height, width = int(image_shape[1]), int(image_shape[2])
        per_image_mb = (height * width * _BYTES_PER_PIXEL_WORKING_SET) / (1024 * 1024)
        if per_image_mb <= 0:
            return max(1, min(_FALLBACK_BATCH_SIZE, count))

        batch = int((total_mb * _USABLE_MEMORY_FRACTION) // per_image_mb)
        batch = min(batch, _MAX_AUTO_BATCH_PIXELS // max(1, height * width))

        # Round down to a power of two: it keeps the driver's tail-padding cheap
        # and gives the allocator a small set of block sizes to reuse.
        batch = 1 << (batch.bit_length() - 1) if batch > 0 else 1
        return max(1, min(batch, count))
