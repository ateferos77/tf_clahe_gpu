#!/usr/bin/env python
"""Reproduce the throughput figures quoted in the README.

    python benchmarks/run_benchmark.py --json results.json

Every number this prints is measured with an explicit device synchronisation,
so it reflects completed GPU work rather than the rate at which kernels can be
queued. See gpu_clahe/benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from gpu_clahe import (
    benchmark_opencv,
    benchmark_performance,
    environment,
    get_gpu_info,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512],
        help="square image sizes to sweep (default: 256 512)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=512,
        help="images processed per timed repeat (default: 512)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="batch sizes to sweep (default: powers of two up to --num-images)",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--clip-limit", type=float, default=0.035)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--skip-opencv", action="store_true", help="skip the CPU baseline"
    )
    parser.add_argument("--json", type=str, default=None, help="write results here")
    return parser.parse_args(argv)


def _command_line(args: argparse.Namespace) -> str:
    """The invocation that reproduces this run, including non-default flags."""
    parts = [
        "python benchmarks/run_benchmark.py",
        f"--sizes {' '.join(str(s) for s in args.sizes)}",
        f"--num-images {args.num_images}",
        f"--repeats {args.repeats}",
        f"--tile-size {args.tile_size}",
        f"--clip-limit {args.clip_limit}",
    ]
    if args.batch_sizes:
        parts.append(f"--batch-sizes {' '.join(str(b) for b in args.batch_sizes)}")
    if args.skip_opencv:
        parts.append("--skip-opencv")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    info = get_gpu_info()
    print(f"TensorFlow {info['tensorflow_version']}  |  GPUs: {info['num_gpus']}")
    for gpu in info["gpus"]:
        print(f"  {gpu['name']}  compute capability {gpu['compute_capability']}")
    if not info["num_gpus"]:
        print("  no GPU visible - the numbers below are CPU timings")
    print()

    sweeps: list[dict[str, object]] = []

    for size in args.sizes:
        print(f"=== {size}x{size} ===")
        report = benchmark_performance(
            image_shape=(size, size),
            num_images=args.num_images,
            batch_sizes=args.batch_sizes,
            tile_size=args.tile_size,
            clip_limit=args.clip_limit,
            num_repeats=args.repeats,
        )
        for result in report.results:
            print(
                f"  batch {result.batch_size:4d}  "
                f"{result.median_s * 1000:8.2f} ms  "
                f"{result.images_per_second:10.1f} img/s  "
                f"{result.megapixels_per_second:8.1f} MPix/s  "
                f"(sd {result.stdev_s * 1000:.2f} ms)"
            )

        best = report.best()
        if best is not None:
            print(
                f"  best: {best.images_per_second:.0f} img/s at batch {best.batch_size}"
            )

        entry: dict[str, object] = {"size": size, "gpu": report.to_dict()}

        if not args.skip_opencv:
            baseline = benchmark_opencv(
                image_shape=(size, size),
                num_images=min(64, args.num_images),
                tile_size=args.tile_size,
                num_repeats=3,
            )
            if baseline is None:
                print("  opencv-python not installed - skipping CPU baseline")
            else:
                print(f"  OpenCV (1 thread): {baseline.images_per_second:.1f} img/s")
                if best is not None:
                    speedup = best.images_per_second / baseline.images_per_second
                    print(f"  speedup: {speedup:.1f}x")
                    entry["speedup_vs_opencv"] = speedup
                entry["opencv"] = asdict(baseline)
        print()
        sweeps.append(entry)

    # Record the invocation, not just the results: the README quotes these
    # numbers, and a reader has to be able to reproduce the exact run rather
    # than guess which flags produced it from the defaults of the day.
    payload = {
        "environment": environment(),
        "parameters": {
            "sizes": args.sizes,
            "num_images": args.num_images,
            "batch_sizes": args.batch_sizes,
            "tile_size": args.tile_size,
            "clip_limit": args.clip_limit,
            "num_repeats": args.repeats,
        },
        "command": _command_line(args),
        "sweeps": sweeps,
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
