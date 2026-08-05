# Changelog

Notable changes to `gpu-clahe`. Versions follow [semantic versioning](https://semver.org/).

## 2.0.0

A rewrite. The package is importable under a new name, the kernel produces different
pixels, and two classes of input that used to be accepted are now rejected. Read the
breaking changes before upgrading.

### Breaking

- **Import name: `clahe` → `gpu_clahe`.** The distribution has always been `gpu-clahe`;
  the module now matches it.

  ```python
  import clahe        # 1.x
  import gpu_clahe    # 2.0
  ```

- **Output pixel values shift by up to one grey level.** The LUT normalization moved from
  float arithmetic with a truncating cast to exact integer round-half-up
  (`(2·255·n + s) // (2·s)`). This is what makes CPU and GPU results bit-identical — XLA:GPU
  lowers float division to an approximate reciprocal, so quotients landing on `x.5` used to
  round differently per device. Pipelines that hash or diff CLAHE output will see a change.

- **Dynamic height and width now raise.** 1.x built the kernel on `tf.shape` and traced with
  unknown spatial dimensions; 2.0 requires `H` and `W` at trace time and raises `ValueError`
  otherwise. Batch size may still be dynamic. Call `set_shape((None, H, W))`, or resize
  before the kernel, or use `clahe_gpu_nojit`.

- **`convert_clahe` validates its input by default.** Values are read on a `[0, 255]` scale,
  and two mis-scaled inputs used to pass silently: a `[0, 1]` normalized float image, which
  comes back as a solid black frame, and a 12-bit integer image, which loses ~94% of its
  pixels to clipping. Both now raise `ValueError`. Pass `validate=False` for the old
  behaviour. `clahe_gpu` is unchanged and still does not check.

- **`use_pipeline=True` is now honoured for every dataset size.** 1.x silently ignored the
  flag below 1000 images (`if use_pipeline and total_images > 1000`).

- **`batch_size` defaults to `None`** — derived from available VRAM, the image size and
  `tile_size` — rather than a fixed `128`. Pass an explicit value to pin it.

- **`setup_gpu()` returns `bool`** (whether a GPU is visible) instead of `None`, and no
  longer swallows every exception through a bare `except:`.

### Added

- `dtype` is plumbed through `convert_clahe`; output is no longer always `uint8`.
- `CLAHEConfig` is accepted by `convert_clahe` and actually takes effect. Explicit
  arguments win over it, including ones that equal the default.
- `clahe_gpu_nojit`, a non-XLA twin for dynamic shapes and for diagnosing a suspected
  miscompile.
- `validate_input` / `require_valid_input`, `get_gpu_info`, `environment`,
  `total_gpu_memory_mb`, `gpu_driver_version`.
- `benchmark_performance` / `benchmark_opencv`, a synchronized harness. Results record the
  parameters that produced them, so a serialized benchmark can be checked against the claim
  it is quoted to support.
- `py.typed` — the package ships type information.

### Fixed

- **Border smearing.** The upper interpolation index was derived from the already-clamped
  lower index, so the outer half-tile blended tiles 0 and 1 using a weight computed for
  tiles −1 and 0. The symptom was a visible seam along the top and left edges.
- **CPU/GPU divergence.** See the LUT change above.
- **Images smaller than half a tile.** `tf.pad` rejects a SYMMETRIC pad wider than the axis
  it pads, which such images require; padding is now applied in chunks. Previously this
  aborted inside the XLA MirrorPad kernel.
- **`clip_limit` small enough to floor the clip value at 0** zeroed every bin and degraded
  the LUT to a linear ramp. The clip value now has a floor of 1.
- **Float input to `convert_clahe`** raised, because the driver forced everything through
  `tf.convert_to_tensor(..., dtype=tf.uint8)`.
- **Batch sizing ignored `tile_size`**, while the histogram buffers scale as
  `1/tile_size²` — 3 B/pixel at 32 but 192 at 4. On a small card this chose a batch that
  did not fit.

### Performance

- Histogram rewritten as a single scatter-add (`unsorted_segment_sum`) instead of a
  256-wide broadcast comparison: 13.3 ms → 1.1 ms at 32×512×512 on a GTX 1650. Since the
  histogram was ~85% of kernel time, this sets the throughput of the package.
- LUT application uses a combined `tile · 256 + value` scalar gather instead of gathering
  each pixel's whole 256-entry LUT, which materialized a `(B, H, W, 256)` tensor — 34 GB
  for a 128×512×512 batch, and the reason 1.x needed tiny batches.
- LUTs are `uint8` rather than `int32`, quartering memory traffic through the gathers.
- Interpolation reads the unpadded image rather than padding and cropping.

Measured: 14,147 img/s at 512×512 (was ~3,252), ~15× single-threaded OpenCV.

### Infrastructure

- `pyproject.toml` is the single source of packaging truth; `setup.py` removed.
- 185 tests, including bit-exact comparison against a NumPy reference implementation.
- CI across Python 3.9–3.13 on Linux, macOS and Windows; release to PyPI via trusted
  publishing.

## 1.0.0

Initial release.