# API reference

Complete reference for `gpu_clahe` 2.0.x. For the algorithm, parameter
conventions and benchmarks, see the [README](https://github.com/ateferos77/tf_clahe_gpu#readme).

```bash
pip install gpu-clahe
```

```python
import gpu_clahe
```

---

## Contents

- [At a glance](#at-a-glance)
- [Processing](#processing) — [`convert_clahe`](#convert_clahe) · [`clahe_gpu`](#clahe_gpu) · [`clahe_gpu_nojit`](#clahe_gpu_nojit)
- [Configuration](#configuration) — [`CLAHEConfig`](#claheconfig) · [`setup_gpu`](#setup_gpu)
- [Validation](#validation) — [`validate_input`](#validate_input) · [`require_valid_input`](#require_valid_input)
- [Environment](#environment-1) — [`environment`](#environment-2) · [`get_gpu_info`](#get_gpu_info) · [`total_gpu_memory_mb`](#total_gpu_memory_mb) · [`gpu_driver_version`](#gpu_driver_version)
- [Benchmarking](#benchmarking) — [`benchmark_performance`](#benchmark_performance) · [`benchmark_opencv`](#benchmark_opencv) · [`sync`](#sync)
- [Input requirements](#input-requirements) · [Errors](#errors) · [Recipes](#recipes)

---

## At a glance

```python
import numpy as np
import gpu_clahe

images = np.random.randint(0, 256, (500, 512, 512), dtype=np.uint8)
enhanced = gpu_clahe.convert_clahe(images)          # (500, 512, 512) uint8
```

| Symbol | Kind | Use it when |
|---|---|---|
| `convert_clahe` | function | **Start here.** NumPy array in, NumPy array out, batching handled |
| `clahe_gpu` | tf.function | You already have a `tf.Tensor` on the device |
| `clahe_gpu_nojit` | tf.function | Shapes cannot be static, or you suspect an XLA miscompile |
| `CLAHEConfig` | dataclass | You want to bundle parameters and reuse them |
| `setup_gpu` | function | You want to configure the device explicitly |
| `validate_input` | function | You want to check input without raising |
| `require_valid_input` | function | You want it to raise |
| `environment` | function | You are recording what a result was measured on |
| `benchmark_performance` | function | You are measuring throughput |

---

## Processing

### `convert_clahe`

```python
convert_clahe(
    images,
    batch_size=None,
    tile_size=None,
    clip_limit=None,
    use_pipeline=False,
    return_tensor=False,
    dtype=None,
    config=None,
    validate=True,
) -> np.ndarray | tf.Tensor
```

The main entry point. Takes a whole dataset, splits it into batches that fit in
GPU memory, and returns the result in the same container shape you gave it.

| Parameter | Default | Meaning |
|---|---|---|
| `images` | — | `(N, H, W)` or `(N, H, W, 1)` array or tensor. Values read on a `[0, 255]` scale whatever the dtype |
| `batch_size` | `None` | Images per GPU call. `None` derives one from VRAM and image size |
| `tile_size` | `None` | Tile side **in pixels**. `None` → from `config`, else 32 |
| `clip_limit` | `None` | Contrast cap as a **fraction of tile pixels**. `None` → from `config`, else 0.035 |
| `use_pipeline` | `False` | Route batches through `tf.data`. Slower for arrays already in RAM |
| `return_tensor` | `False` | Return a `tf.Tensor` instead of a NumPy array |
| `dtype` | `None` | Output dtype. `None` → from `config`, else `tf.uint8` |
| `config` | `None` | A `CLAHEConfig` supplying defaults |
| `validate` | `True` | Run `validate_input` first. Costs ~7% on uint8, ~13% on float |

Returns an array (or tensor) of the same shape as `images`.

Raises `ValueError` if `images` is empty, `batch_size` is not positive, or
validation fails.

```python
# defaults
out = gpu_clahe.convert_clahe(images)

# tuned for radiographs with a large dark background
out = gpu_clahe.convert_clahe(images, tile_size=32, clip_limit=0.010)

# stay on the GPU, skip validation in a hot loop
out = gpu_clahe.convert_clahe(images, return_tensor=True, validate=False)
```

**Explicit arguments always win over `config`**, including when they equal the
default — passing `tile_size=32` means 32, even if `config.tile_size` is 16.

---

### `clahe_gpu`

```python
clahe_gpu(images, tile_size=32, clip_limit=0.035, dtype=tf.uint8) -> tf.Tensor
```

The XLA-compiled kernel, wrapped in `@tf.function(jit_compile=True)`. Use it
inside an existing TensorFlow pipeline to avoid a host round trip — which costs
more than the CLAHE itself.

- `images` must be a `tf.Tensor` of rank 3 `(B, H, W)` or rank 4 `(B, H, W, 1)`
- `H` and `W` **must be static**; `B` may be dynamic, but a dynamic batch selects
  a histogram path roughly 12× slower
- Integer `dtype` outputs are rounded; float outputs keep sub-level precision

```python
import tensorflow as tf

batch = tf.cast(tf.random.uniform((64, 512, 512), 0, 256, tf.int32), tf.uint8)
enhanced = gpu_clahe.clahe_gpu(batch, tile_size=32, clip_limit=0.035)
```

Inside a `tf.data` pipeline:

```python
ds = (tf.data.Dataset.from_tensor_slices(images)
        .batch(64, drop_remainder=True)          # drop_remainder keeps shapes static
        .map(gpu_clahe.clahe_gpu, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
```

---

### `clahe_gpu_nojit`

```python
clahe_gpu_nojit(images, tile_size, clip_limit, dtype) -> tf.Tensor
```

Same computation without XLA. All four arguments are positional-or-keyword with
**no defaults**. Two uses: shapes that cannot be made static, and checking
whether a suspicious result is an XLA miscompile.

```python
out = gpu_clahe.clahe_gpu_nojit(batch, 32, 0.035, tf.uint8)
```

It produces identical output to `clahe_gpu`, which the test suite asserts.

---

## Configuration

### `CLAHEConfig`

```python
CLAHEConfig(
    tile_size=32,
    clip_limit=0.035,
    dtype=tf.uint8,
    enable_xla=True,
    memory_growth=True,
)
```

A dataclass bundling parameters. Validates at construction: `tile_size` must be
≥ 2 and `clip_limit` > 0, otherwise `ValueError`.

```python
from gpu_clahe import CLAHEConfig, convert_clahe

config = CLAHEConfig(tile_size=16, clip_limit=0.02)
out = convert_clahe(images, config=config)
```

#### `CLAHEConfig.auto_batch_size`

```python
auto_batch_size(image_shape, tile_size=None) -> int
```

Picks a batch size whose working set fits in VRAM, given a dataset shape
`(N, H, W)` or `(N, H, W, C)`. Returns a value in `[1, N]`, rounded down to a
power of two, falling back to a conservative default when device memory cannot
be determined.

```python
config.auto_batch_size(images.shape)          # uses config.tile_size
config.auto_batch_size(images.shape, tile_size=8)
```

The result depends on `tile_size` because per-tile histograms dominate memory at
small tiles: 3 B/pixel at `tile_size=32`, but 48 at 8 and 192 at 4.

---

### `setup_gpu`

```python
setup_gpu(memory_growth=True, enable_xla=True) -> bool
```

Configures GPU memory growth and XLA. Returns `True` if a GPU is visible.
`convert_clahe` calls this for you; call it directly only if you want the return
value or non-default settings.

Memory growth can only be set before a GPU is initialised — calling this later
leaves the existing setting in place rather than raising.

```python
if not gpu_clahe.setup_gpu():
    print("running on CPU")
```

---

## Validation

### `validate_input`

```python
validate_input(images) -> tuple[bool, str]
```

Returns `(is_valid, message)`. `message` is empty when valid, otherwise explains
what to change. Never raises.

Checks rank (3 or 4), a trailing channel of 1 for 4-D, non-empty, supported
dtype, static `H`/`W` for tensors, and — the common silent failure — a float
image normalised to `[0, 1]`, which would otherwise clamp to the bottom LUT bins
and come back as noise.

```python
ok, why = gpu_clahe.validate_input(images)
if not ok:
    print(why)
    # "Float input peaks at 0.998, which looks like a [0, 1] normalised image.
    #  Values are interpreted on a [0, 255] scale; multiply by 255 first."
```

### `require_valid_input`

```python
require_valid_input(images) -> None
```

The raising counterpart. Raises `ValueError` with the same message.

---

## Environment

### `environment`

```python
environment() -> dict[str, Any]
```

Everything needed to describe where a measurement was taken: TensorFlow version,
CUDA build flag, GPU list, Python version, platform and driver version. Keys:
`tensorflow_version`, `built_with_cuda`, `num_gpus`, `gpus`, `python`,
`platform`, `driver_version`.

```python
import json
print(json.dumps(gpu_clahe.environment(), indent=2))
```

Record this alongside any throughput number you publish — a figure without its
hardware is not a result.

### `get_gpu_info`

```python
get_gpu_info() -> dict[str, Any]
```

The GPU subset: `tensorflow_version`, `built_with_cuda`, `num_gpus`, and per-GPU
`name` and `compute_capability`. Never raises; a probe failure lands in an
`error` key.

### `total_gpu_memory_mb`

```python
total_gpu_memory_mb() -> int
```

Total memory of the first visible GPU in MB, via `nvidia-smi`. Returns `0` when
unavailable — read that as "unknown", not "no memory". TensorFlow exposes only
current and peak usage, never the device total, hence the subprocess.

### `gpu_driver_version`

```python
gpu_driver_version() -> str | None
```

NVIDIA driver version string, or `None` if it cannot be determined.

---

## Benchmarking

### `benchmark_performance`

```python
benchmark_performance(
    image_shape=(512, 512),
    num_images=512,
    batch_sizes=None,
    tile_size=32,
    clip_limit=0.035,
    num_repeats=7,
    warmup_repeats=2,
    seed=0,
) -> BenchmarkReport
```

Sweeps batch sizes on synthetic images and returns a `BenchmarkReport`. Batches
are staged on the device before timing, warmup passes absorb XLA compilation, and
every timed region ends with a device sync — so the figure is compute, not
dispatch.

```python
report = gpu_clahe.benchmark_performance(image_shape=(256, 256), num_images=1024)
best = report.best()
print(f"{best.images_per_second:,.0f} img/s at batch {best.batch_size}")
```

**`BenchmarkReport`** has `environment: dict`, `results: list[BenchmarkResult]`,
`.best()` returning the fastest result or `None`, and `.to_dict()` for
`json.dump`.

**`BenchmarkResult`** (importable from `gpu_clahe.benchmark`) carries
`batch_size`, `num_images`, `image_shape`, `tile_size`, `clip_limit`,
`num_repeats`, `median_s`, `mean_s`, `stdev_s`, `min_s`, `images_per_second`,
`megapixels_per_second`. The **median** is reported, so one scheduling hiccup
does not skew the number.

### `benchmark_opencv`

```python
benchmark_opencv(
    image_shape=(512, 512), num_images=64, tile_size=32,
    clip_limit=2.0, num_repeats=3, seed=0,
) -> BenchmarkResult | None
```

Single-threaded OpenCV baseline, or `None` if OpenCV is not installed. Note
`clip_limit` here is **OpenCV's** scale, and `tileGridSize` is derived from
`tile_size` and the image dimensions so the comparison is like-for-like.

### `sync`

```python
from gpu_clahe.benchmark import sync
sync()
```

Blocks until queued device work has finished. **Call this before stopping any
clock.** TensorFlow returns as soon as work is *queued*, so an unsynchronised
timing loop measures dispatch rate and reports numbers an order of magnitude too
high.

---

## Input requirements

| Requirement | Detail |
|---|---|
| **Shape** | `(N, H, W)` or `(N, H, W, 1)` |
| **Channels** | Single channel only. For colour, convert to LAB and apply to L |
| **Scale** | Values read on `[0, 255]` **whatever the dtype** |
| **H, W** | Must be static at trace time; `N` may be dynamic |
| **Dtypes** | `uint8`, `uint16`, `int16`, `int32`, `float16`, `float32`, `float64` |

⚠️ **Out-of-range values are clipped, not rescaled.** A 12-bit DICOM radiograph
(0–4095) passes validation and then loses 93.8% of its range. Rescale first:

```python
scaled = image.astype(np.float32) / image.max() * 255
```

---

## Errors

| Message | Cause | Fix |
|---|---|---|
| `Cannot process an empty image array.` | `len(images) == 0` | — |
| `batch_size must be positive, got …` | non-positive `batch_size` | — |
| `Only single-channel images are supported; got N channels.` | RGB input | convert to grayscale, or apply per channel |
| `clahe_gpu requires static spatial dimensions` | dynamic `H`/`W` | `set_shape((None, H, W))`, or resize/`padded_batch` first |
| `Expected a 3-D (B, H, W) or 4-D (B, H, W, 1) tensor, got rank N.` | wrong rank | add or drop the batch axis |
| `tile_size must be >= 2, got N.` | tile too small | — |
| `clip_limit must be > 0, got N.` | non-positive clip | — |
| `Float input peaks at 0.998, which looks like a [0, 1] normalised image.` | normalised floats | multiply by 255 |

---

## Recipes

**Colour images, via the L channel of LAB**

```python
import cv2, numpy as np, gpu_clahe

lab = np.stack([cv2.cvtColor(i, cv2.COLOR_RGB2LAB) for i in rgb_images])
lab[..., 0] = gpu_clahe.convert_clahe(lab[..., 0])
out = np.stack([cv2.cvtColor(i, cv2.COLOR_LAB2RGB) for i in lab])
```

**Matching an existing OpenCV pipeline**

```python
# OpenCV: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)) on 512x512
#         clip_limit = clipLimit / 256 ;  tile_size = width / grid_cols
out = gpu_clahe.convert_clahe(images, tile_size=32, clip_limit=2.0 / 256)
```

**High-bit-depth (DICOM) input**

```python
scaled = images.astype(np.float32) / images.max() * 255
out = gpu_clahe.convert_clahe(scaled, dtype=tf.uint8)
```

**Maximum throughput — keep data on the device**

```python
import tensorflow as tf
from gpu_clahe.benchmark import sync

device_batch = tf.constant(images[:128])
out = gpu_clahe.clahe_gpu(device_batch)     # no host round trip
sync()                                       # only if you are timing it
```

**Reproducible preprocessing**

Output is bit-identical across CPU/GPU, Python 3.9–3.13, TensorFlow 2.20–2.21
and NumPy 2.0–2.5. Nothing to configure — it is a property of the integer
lookup-table arithmetic. Record `gpu_clahe.environment()` with your results
anyway.
