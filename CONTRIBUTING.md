# Contributing

## Setup

```bash
git clone https://github.com/ateferos77/tf_clahe_gpu
cd tf_clahe_gpu
pip install -e ".[dev]"
pre-commit install
```

Add `pip install "tensorflow[and-cuda]"` for a GPU build. Everything works on
CPU; GPU-marked tests skip automatically when no device is visible.

## Checks

CI runs exactly these, so run them before pushing:

```bash
pytest
ruff check .
ruff format --check .
mypy gpu_clahe
```

## Changing the kernel

`tests/reference.py` is a naive NumPy implementation that mirrors
`gpu_clahe/core.py` step for step. The test suite asserts the two are
**bit-identical**, which is what makes regressions obvious rather than
"probably still fine".

That means any change to the algorithm has to be made in both places. If you
change one and the tests fail, the reference is the one to trust — it is written
to be read, not to be fast.

Two properties are load-bearing and easy to break:

- **Determinism across devices.** Keep the LUT normalization in integer
  arithmetic. Float division is not portable here: XLA:GPU lowers it to an
  approximate reciprocal, so quotients that land exactly on `x.5` round
  differently on CPU and GPU. `test_gpu_matches_cpu_bit_exactly` guards this.
- **Determinism under atomics.** The histogram scatter accumulates int32 ones.
  Integer addition is associative, so scatter ordering cannot change the result.
  Accumulating in float would silently make output run-dependent.

## Performance work

Benchmark before and after, on the same machine, and quote the hardware:

```bash
python benchmarks/run_benchmark.py --sizes 256 512 1024
```

`gpu_clahe.benchmark.sync()` must be called before stopping any clock.
TensorFlow returns as soon as work is *queued*, so an unsynchronized timing loop
measures dispatch rate and will happily report numbers an order of magnitude too
high. This is how the pre-2.0 benchmark produced figures nothing could
reproduce.

## Releasing

1. Update `__version__` in `gpu_clahe/version.py`.
2. Add a `CHANGELOG.md` entry. Anything that changes output pixels, rejects input that
   used to be accepted, or moves a default belongs under **Breaking** — those are the
   changes a user cannot discover from the API surface.
3. Tag `vX.Y.Z`. The release workflow checks the tag matches `__version__`, builds, and
   publishes to PyPI via trusted publishing.
