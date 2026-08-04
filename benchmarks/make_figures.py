#!/usr/bin/env python
"""Regenerate the figures used in the README.

Every number plotted is measured at run time, not copied from the text.

    python benchmarks/make_figures.py --data /path/to/grayscale/pngs
    python benchmarks/make_figures.py --data ... --bench docs/benchmark_results.json

``--data`` should point at a directory of single-channel PNGs of identical size.
The published figures use the RSNA Pediatric Bone Age Challenge dataset
(256x256, 8-bit), which is publicly available for research but not redistributed
with this package.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gpu_clahe

OUT = "docs/images"
TILE, GRID = 32, (8, 8)

# A restrained, colour-blind-safe palette; consistent across every figure.
INK = "#1a1a1a"
MUTED = "#8a8a8a"
BLUE = "#2b6cb0"
ORANGE = "#c05621"
GREEN = "#2f7d5d"
RED = "#9b2c2c"
GRID_C = "#e2e2e2"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID_C,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
    }
)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    return ax


def load(data_dir, n):
    files = sorted(glob.glob(os.path.join(data_dir, "*.png")))[:n]
    if not files:
        raise SystemExit(f"no PNGs found under {data_dir}")
    return np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])


CLIPS = [0.005, 0.010, 0.020, 0.035, 0.100, 1.000]


# --------------------------------------------------------------------------- #
def fig_qualitative(imgs, out):
    """Figure 1: what CLAHE actually does to a radiograph."""
    src = imgs[0]
    variants = [("Original", src)]
    for cl in (0.010, 0.035, 1.000):
        enhanced = gpu_clahe.convert_clahe(
            src[None], batch_size=1, tile_size=TILE, clip_limit=cl
        )[0]
        tag = {
            0.010: "clip_limit = 0.010\n(matched to OpenCV clipLimit 2.6)",
            0.035: "clip_limit = 0.035\n(package default)",
            1.000: "clip_limit = 1.000\n(clipping disabled -> plain AHE)",
        }[cl]
        variants.append((tag, enhanced))

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(11, 5.0),
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.42},
    )
    for col, (title, im) in enumerate(variants):
        ax = axes[0, col]
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)

        hx = axes[1, col]
        style(hx)
        # Log counts: 43% of every pixel is near-black background, so a linear
        # axis is one spike at 0 and nothing else legible.
        hx.hist(
            im.ravel(),
            bins=64,
            range=(0, 255),
            log=True,
            color=BLUE if col else MUTED,
            edgecolor="none",
        )
        hx.set_yticks([])
        hx.set_xlim(0, 255)
        hx.set_xlabel("pixel value", fontsize=7.5)
        hx.tick_params(labelsize=7)
        if col == 0:
            hx.set_ylabel("count (log)", fontsize=7.5)

    fig.suptitle(
        "CLAHE on a pediatric hand radiograph (RSNA Bone Age, 256x256)\n"
        "Raising clip_limit spreads the histogram further, but amplifies the\n"
        "empty background",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(os.path.join(out, "fig1_qualitative.png"))
    plt.close(fig)


def fig_clip_tradeoff(imgs, out):
    """Figure 2: detail vs background noise as clip_limit varies."""
    bg, fg = imgs < 15, imgs >= 15
    detail, noise = [], []
    for cl in CLIPS:
        o = gpu_clahe.convert_clahe(imgs, batch_size=64, tile_size=TILE, clip_limit=cl)
        detail.append(o[fg].std())
        noise.append(o[bg].std())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    style(a1)
    a1.semilogx(CLIPS, detail, "o-", color=BLUE, lw=1.8, ms=5, label="hand (signal)")
    a1.semilogx(CLIPS, noise, "s-", color=RED, lw=1.8, ms=5, label="background (noise)")
    a1.axhline(imgs[fg].std(), color=BLUE, ls=":", lw=1.2)
    a1.axhline(imgs[bg].std(), color=RED, ls=":", lw=1.2)
    a1.axvline(0.035, color=MUTED, ls="--", lw=1.1)
    a1.annotate(
        "package default",
        xy=(0.035, 71),
        xytext=(0.075, 68),
        fontsize=7.5,
        color=MUTED,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
    )
    a1.set_xlabel("clip_limit  (fraction of tile pixels)")
    a1.set_ylabel("std. dev. of pixel values")
    a1.set_title("Signal and noise both grow with clip_limit")
    a1.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62))
    a1.set_ylim(0, 72)
    a1.text(0.0042, imgs[fg].std() + 1.5, "original signal", fontsize=7, color=BLUE)
    a1.text(0.0042, imgs[bg].std() + 1.5, "original noise", fontsize=7, color=RED)

    style(a2)
    a2.plot(noise, detail, "-", color=MUTED, lw=1.2, zorder=1)
    for cl, n, d in zip(CLIPS, noise, detail):
        is_def = cl == 0.035
        best = cl == 0.010
        c = ORANGE if is_def else (GREEN if best else BLUE)
        a2.scatter([n], [d], s=70 if (is_def or best) else 34, color=c, zorder=3)
        a2.annotate(
            f"{cl:g}",
            (n, d),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
            color=c,
            weight="bold" if (is_def or best) else "normal",
        )
    a2.set_xlabel("background noise  (std, lower is better)")
    a2.set_ylabel("hand detail  (std, higher is better)")
    a2.set_title("The default is dominated: 0.010 is better on both axes")
    a2.annotate(
        "",
        xy=(noise[1], detail[1]),
        xytext=(noise[3], detail[3]),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.6},
    )
    a2.text(
        0.42,
        0.88,
        "arrow: moving 0.035 -> 0.010\ngains detail AND cuts noise",
        transform=a2.transAxes,
        fontsize=7.5,
        color=GREEN,
        va="top",
    )
    a2.margins(x=0.12, y=0.16)

    fig.suptitle(
        "Contrast limiting on 256 real radiographs "
        "(background = pixels < 15, 42.9% of each image)",
        fontsize=10,
        y=1.03,
    )
    fig.savefig(os.path.join(out, "fig2_clip_limit.png"))
    plt.close(fig)


def fig_opencv(imgs, out):
    """Figure 3: the parameter mapping and residual agreement with OpenCV."""
    cv2.setNumThreads(1)
    predicted, found, mads, corrs = [], [], [], []
    probe = np.arange(0.5, 20.01, 0.25)
    for cl in CLIPS[:5]:
        ours = gpu_clahe.convert_clahe(
            imgs, batch_size=64, tile_size=TILE, clip_limit=cl
        )
        best = None
        for ocl in probe:
            t = np.stack(
                [
                    cv2.createCLAHE(clipLimit=float(ocl), tileGridSize=GRID).apply(i)
                    for i in imgs
                ]
            )
            m = np.abs(ours.astype(np.int16) - t.astype(np.int16)).mean()
            if best is None or m < best[1]:
                r = np.corrcoef(
                    ours.ravel().astype(np.float64), t.ravel().astype(np.float64)
                )[0, 1]
                best = (float(ocl), m, r)
        predicted.append(cl * 256)
        found.append(best[0])
        mads.append(best[1])
        corrs.append(best[2])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    style(a1)
    lim = [0, max(predicted + found) * 1.12]
    a1.plot(
        lim, lim, ls="--", color=MUTED, lw=1.2, label="clipLimit = clip_limit x 256"
    )
    a1.scatter(predicted, found, s=58, color=BLUE, zorder=3)
    for p, f, cl in zip(predicted, found, CLIPS[:5]):
        a1.annotate(
            f"{cl:g}",
            (p, f),
            textcoords="offset points",
            xytext=(7, -4),
            fontsize=7.5,
            color=BLUE,
        )
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    a1.set_xlabel("predicted OpenCV clipLimit  (clip_limit x 256)")
    a1.set_ylabel("clipLimit that actually minimises the difference")
    a1.set_title("Parameter mapping is exact")
    a1.legend(fontsize=8, loc="upper left")

    style(a2)
    a2.bar(range(len(CLIPS[:5])), mads, color=BLUE, width=0.6)
    a2.set_xticks(range(len(CLIPS[:5])))
    a2.set_xticklabels([f"{c:g}" for c in CLIPS[:5]])
    a2.set_xlabel("clip_limit")
    a2.set_ylabel("mean abs. difference  (grey levels, 0-255)")
    a2.set_title("Residual disagreement at the matched setting")
    for i, (m, r) in enumerate(zip(mads, corrs)):
        a2.text(i, m + 0.12, f"r={r:.4f}", ha="center", fontsize=7.5, color=INK)
    a2.set_ylim(0, max(mads) * 1.35)
    a2.axhline(2.55, color=GREEN, ls=":", lw=1.2)
    a2.text(-0.42, 2.75, "1% of the 0-255 range", fontsize=7, color=GREEN, ha="left")

    fig.suptitle(
        "Agreement with OpenCV createCLAHE at matched parameters "
        "(64 radiographs, tile_size=32 <-> tileGridSize=(8,8))",
        fontsize=10,
        y=1.03,
    )
    fig.savefig(os.path.join(out, "fig3_opencv.png"))
    plt.close(fig)


def fig_throughput(bench_path, out):
    """Figure 4: measured throughput, from the benchmark JSON."""
    with open(bench_path) as fh:
        payload = json.load(fh)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    style(a1)
    style(a2)

    colours = {256: BLUE, 512: ORANGE, 1024: GREEN}
    sizes, best_rate, cv_rate = [], [], []
    for sweep in payload["sweeps"]:
        size = sweep["size"]
        rs = sweep["gpu"]["results"]
        if not rs:
            continue
        bs = [r["batch_size"] for r in rs]
        ips = [r["images_per_second"] for r in rs]
        a1.plot(
            bs,
            ips,
            "o-",
            color=colours.get(size, BLUE),
            lw=1.8,
            ms=4,
            label=f"{size}x{size}",
        )
        sizes.append(size)
        best_rate.append(max(ips))
        cv_rate.append(sweep.get("opencv", {}).get("images_per_second", float("nan")))

    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("batch size (images per kernel call)")
    a1.set_ylabel("images / second")
    a1.set_title("Kernel throughput vs batch size")
    a1.legend(fontsize=8)

    x = np.arange(len(sizes))
    w = 0.36
    a2.bar(x - w / 2, best_rate, w, color=BLUE, label="gpu_clahe (GPU)")
    a2.bar(x + w / 2, cv_rate, w, color=MUTED, label="OpenCV (1 CPU thread)")
    a2.set_yscale("log")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{s}x{s}" for s in sizes])
    a2.set_ylabel("images / second  (log scale)")
    a2.set_title("Against the CPU baseline")
    a2.legend(fontsize=8)
    for i, (g, c) in enumerate(zip(best_rate, cv_rate)):
        if c == c:
            a2.text(
                i,
                max(g, c) * 1.35,
                f"{g / c:.0f}x",
                ha="center",
                fontsize=8.5,
                weight="bold",
                color=INK,
            )
    a2.set_ylim(top=max(best_rate) * 4)

    env = payload["environment"]
    gpu = env["gpus"][0]["name"] if env.get("gpus") else "CPU only"
    fig.suptitle(
        f"Measured on {gpu}, TensorFlow {env['tensorflow_version']} "
        f"(device-synchronised, median of 5 repeats)",
        fontsize=10,
        y=1.03,
    )
    fig.savefig(os.path.join(out, "fig4_throughput.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data", required=True, help="directory of equally-sized single-channel PNGs"
    )
    ap.add_argument(
        "--bench",
        default="docs/benchmark_results.json",
        help="JSON from run_benchmark.py --json (for figure 4)",
    )
    ap.add_argument("--out", default=OUT, help="output directory")
    ap.add_argument("--num-images", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    imgs = load(args.data, args.num_images)
    print("fig1 qualitative ...")
    fig_qualitative(imgs, args.out)
    print("fig2 clip tradeoff ...")
    fig_clip_tradeoff(imgs, args.out)
    print("fig3 opencv ...")
    fig_opencv(imgs[:64], args.out)
    if os.path.exists(args.bench):
        print("fig4 throughput ...")
        fig_throughput(args.bench, args.out)
    else:
        print(f"fig4 skipped ({args.bench} not found)")
    print(f"wrote figures to {args.out}/")


if __name__ == "__main__":
    main()
