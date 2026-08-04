"""The benchmark harness itself.

A benchmark that silently measures the wrong thing is worse than none, so the
harness gets tested like any other code - especially the synchronisation, which
is what the previous version got wrong.
"""

from __future__ import annotations

import json

import pytest

from gpu_clahe import benchmark_opencv, benchmark_performance
from gpu_clahe.benchmark import sync


def test_sync_is_callable_on_any_backend() -> None:
    """Must be a no-op rather than an error when there is no GPU."""
    sync()


@pytest.fixture(scope="module")
def small_report():
    return benchmark_performance(
        image_shape=(64, 64),
        num_images=8,
        batch_sizes=[4, 8],
        num_repeats=2,
        warmup_repeats=1,
    )


def test_reports_one_result_per_batch_size(small_report) -> None:
    assert [r.batch_size for r in small_report.results] == [4, 8]


def test_timings_are_positive_and_consistent(small_report) -> None:
    for result in small_report.results:
        assert result.median_s > 0
        assert result.min_s <= result.median_s
        assert result.stdev_s >= 0
        # Throughput must be derived from the median, not invented.
        assert result.images_per_second == pytest.approx(
            result.num_images / result.median_s
        )


def test_megapixel_rate_matches_the_image_rate(small_report) -> None:
    for result in small_report.results:
        pixels = result.image_shape[0] * result.image_shape[1]
        assert result.megapixels_per_second == pytest.approx(
            result.images_per_second * pixels / 1e6
        )


def test_batch_sizes_larger_than_the_dataset_are_skipped() -> None:
    report = benchmark_performance(
        image_shape=(32, 32),
        num_images=4,
        batch_sizes=[4, 64],
        num_repeats=1,
        warmup_repeats=1,
    )
    assert [r.batch_size for r in report.results] == [4]


def test_environment_is_recorded(small_report) -> None:
    """A throughput number without its hardware is not a result."""
    environment = small_report.environment
    assert environment["tensorflow_version"]
    assert "num_gpus" in environment
    assert environment["platform"]


def test_report_is_json_serialisable(small_report) -> None:
    payload = json.dumps(small_report.to_dict())
    assert json.loads(payload)["results"]


def test_best_picks_the_highest_throughput(small_report) -> None:
    best = small_report.best()
    assert best is not None
    assert best.images_per_second == max(
        r.images_per_second for r in small_report.results
    )


@pytest.mark.parametrize(("num_images", "num_repeats"), [(0, 1), (4, 0)])
def test_invalid_arguments_are_rejected(num_images: int, num_repeats: int) -> None:
    with pytest.raises(ValueError):
        benchmark_performance(
            image_shape=(32, 32), num_images=num_images, num_repeats=num_repeats
        )


def test_opencv_baseline_runs_or_reports_absence() -> None:
    result = benchmark_opencv(image_shape=(64, 64), num_images=4, num_repeats=1)
    if result is None:
        pytest.skip("opencv-python not installed")
    assert result.images_per_second > 0
    assert result.median_s > 0
