"""Throughput measurement.

The one thing that makes or breaks a GPU benchmark is synchronisation.
TensorFlow enqueues work on the device and returns a handle immediately, so a
timing loop that never forces completion measures the *dispatch* rate, not the
*compute* rate, and reports numbers that are wrong by an order of magnitude.
Every measurement here ends with an explicit device sync.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from .core import clahe_gpu, setup_gpu
from .utils import get_gpu_info

__all__ = [
    "BenchmarkResult",
    "benchmark_opencv",
    "benchmark_performance",
    "environment",
    "sync",
]


def sync() -> None:
    """Block until all queued device work has finished.

    ``sync_devices`` is the cheap, transfer-free way to do this. It landed in
    TF 2.12; on older builds we fall back to a one-element device-to-host copy,
    which also drains the stream.
    """
    try:
        tf.test.experimental.sync_devices()
    except (AttributeError, RuntimeError):
        if tf.config.list_physical_devices("GPU"):
            with tf.device("/GPU:0"):
                tf.constant(0).numpy()


@dataclass
class BenchmarkResult:
    """Timings for a single configuration.

    Attributes:
        batch_size: Images per kernel call.
        num_images: Total images processed per repeat.
        image_shape: ``(H, W)`` of each image.
        median_s: Median wall time of one full pass over ``num_images``.
        mean_s: Mean wall time.
        stdev_s: Standard deviation across repeats.
        min_s: Fastest repeat, i.e. the best case with a warm cache.
        images_per_second: Throughput derived from ``median_s``.
        megapixels_per_second: Shape-independent throughput.
    """

    batch_size: int
    num_images: int
    image_shape: Sequence[int]
    median_s: float
    mean_s: float
    stdev_s: float
    min_s: float
    images_per_second: float
    megapixels_per_second: float


@dataclass
class BenchmarkReport:
    """A full benchmark run plus the environment it was measured in."""

    environment: dict[str, Any]
    results: list[BenchmarkResult] = field(default_factory=list)

    def best(self) -> BenchmarkResult | None:
        """The configuration with the highest throughput, or None if empty."""
        return max(self.results, key=lambda r: r.images_per_second, default=None)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for ``json.dump``."""
        return {
            "environment": self.environment,
            "results": [asdict(r) for r in self.results],
        }


def environment() -> dict[str, Any]:
    """GPU/TF info plus the Python version and platform.

    Public because a throughput number is only meaningful alongside the machine
    that produced it, so anything writing a benchmark artefact should record
    this rather than the narrower :func:`gpu_clahe.get_gpu_info`.
    """
    info = get_gpu_info()
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


def _time_pass(fn: Callable[[tf.Tensor], Any], batches: Sequence[tf.Tensor]) -> float:
    """Wall time for one pass over pre-staged batches, synchronised at the end.

    Batches are uploaded before timing starts so this measures compute rather
    than PCIe bandwidth; :func:`benchmark_performance` reports the end-to-end
    figure separately.
    """
    start = time.perf_counter()
    for batch in batches:
        fn(batch)
    sync()
    return time.perf_counter() - start


def benchmark_performance(
    image_shape: Sequence[int] = (512, 512),
    num_images: int = 512,
    batch_sizes: Sequence[int] | None = None,
    tile_size: int = 32,
    clip_limit: float = 0.035,
    num_repeats: int = 7,
    warmup_repeats: int = 2,
    seed: int = 0,
) -> BenchmarkReport:
    """Measure kernel throughput across batch sizes.

    Args:
        image_shape: ``(H, W)`` of each test image.
        num_images: Images processed per repeat.
        batch_sizes: Batch sizes to sweep. Defaults to powers of two up to
            ``num_images``.
        tile_size: CLAHE tile side length.
        clip_limit: Contrast limit as a fraction of tile pixels.
        num_repeats: Timed repeats per batch size. The median is reported, so
            an odd count avoids interpolation.
        warmup_repeats: Untimed passes run first, to absorb XLA compilation and
            let the GPU clock up. Never fewer than 1.
        seed: Seed for the synthetic test images.

    Returns:
        A :class:`BenchmarkReport` with one entry per batch size that fits.

    Raises:
        ValueError: If ``num_images`` or ``num_repeats`` is not positive.
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}.")
    if num_repeats <= 0:
        raise ValueError(f"num_repeats must be positive, got {num_repeats}.")

    setup_gpu()
    height, width = int(image_shape[0]), int(image_shape[1])

    if batch_sizes is None:
        batch_sizes = [b for b in (8, 16, 32, 64, 128, 256) if b <= num_images]

    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(num_images, height, width), dtype=np.uint8)

    report = BenchmarkReport(environment=environment())

    for batch_size in batch_sizes:
        if batch_size > num_images:
            continue

        # Stage every batch on the device up front. Leftover images beyond a
        # whole number of batches are dropped so each repeat does identical work.
        num_batches = num_images // batch_size
        if num_batches == 0:
            continue
        try:
            batches = [
                tf.constant(images[i * batch_size : (i + 1) * batch_size])
                for i in range(num_batches)
            ]
        except tf.errors.ResourceExhaustedError:
            # Larger batch sizes will fail too, so stop rather than continue.
            break

        def run(batch: tf.Tensor) -> tf.Tensor:
            return clahe_gpu(batch, tile_size=tile_size, clip_limit=clip_limit)

        try:
            for _ in range(max(1, warmup_repeats)):
                _time_pass(run, batches)
            timings = [_time_pass(run, batches) for _ in range(num_repeats)]
        except tf.errors.ResourceExhaustedError:
            del batches
            break

        processed = num_batches * batch_size
        median = statistics.median(timings)
        pixels = processed * height * width

        report.results.append(
            BenchmarkResult(
                batch_size=batch_size,
                num_images=processed,
                image_shape=(height, width),
                median_s=median,
                mean_s=statistics.fmean(timings),
                stdev_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
                min_s=min(timings),
                images_per_second=processed / median,
                megapixels_per_second=pixels / median / 1e6,
            )
        )
        del batches

    return report


def benchmark_opencv(
    image_shape: Sequence[int] = (512, 512),
    num_images: int = 64,
    tile_size: int = 32,
    clip_limit: float = 2.0,
    num_repeats: int = 3,
    seed: int = 0,
) -> BenchmarkResult | None:
    """Measure single-threaded OpenCV CLAHE for a reference point.

    Two parameterisation differences are worth stating, since they are the usual
    source of bogus speedup ratios:

    * OpenCV's ``tileGridSize`` is the *number of tiles*, not their pixel size,
      so it is derived here from ``tile_size`` and the image dimensions.
    * OpenCV's ``clipLimit`` is a multiplier on the mean bin height, not a
      fraction of tile pixels, so it is not numerically comparable to this
      package's ``clip_limit``. This measures the cost of a *typical* OpenCV
      call, which is the right thing for a speedup ratio to compare against.

    Returns:
        A :class:`BenchmarkResult`, or ``None`` if OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError:
        return None

    height, width = int(image_shape[0]), int(image_shape[1])
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(num_images, height, width), dtype=np.uint8)

    # tileGridSize is (columns, rows).
    grid = (max(1, -(-width // tile_size)), max(1, -(-height // tile_size)))

    # OpenCV parallelises internally; pin it to one thread so the number is a
    # per-core baseline rather than a property of this machine's core count.
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        for image in images[: min(4, num_images)]:  # warm up
            clahe.apply(image)

        timings = []
        for _ in range(num_repeats):
            start = time.perf_counter()
            for image in images:
                clahe.apply(image)
            timings.append(time.perf_counter() - start)
    finally:
        cv2.setNumThreads(previous_threads)

    median = statistics.median(timings)
    return BenchmarkResult(
        batch_size=1,
        num_images=num_images,
        image_shape=(height, width),
        median_s=median,
        mean_s=statistics.fmean(timings),
        stdev_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_s=min(timings),
        images_per_second=num_images / median,
        megapixels_per_second=num_images * height * width / median / 1e6,
    )
