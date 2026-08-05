"""Shared fixtures and test configuration."""

from __future__ import annotations

import os

# Quieten TF's startup banner before it is imported, so test output stays
# readable. Must happen ahead of the first `import tensorflow`.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest
import tensorflow as tf


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a visible GPU")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip GPU-marked tests when no GPU is visible (e.g. in CI).

    pytest injects hook arguments by name, so unused ones are simply omitted.
    """
    if tf.config.list_physical_devices("GPU"):
        return
    skip = pytest.mark.skip(reason="no GPU visible")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    return bool(tf.config.list_physical_devices("GPU"))


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so any failure reproduces exactly."""
    return np.random.default_rng(20240514)


@pytest.fixture
def uniform_noise(rng: np.random.Generator) -> np.ndarray:
    """Worst case for CLAHE: a flat histogram with nothing to equalise."""
    return rng.integers(0, 256, size=(3, 64, 96), dtype=np.uint8)


@pytest.fixture
def low_contrast(rng: np.random.Generator) -> np.ndarray:
    """The case CLAHE exists for: everything crammed into a narrow band."""
    base = rng.normal(loc=128.0, scale=6.0, size=(3, 64, 96))
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def gradient() -> np.ndarray:
    """A smooth ramp - makes tile-boundary artefacts obvious."""
    row = np.linspace(0, 255, 96, dtype=np.float32)
    column = np.linspace(0, 255, 64, dtype=np.float32)
    image = (row[None, :] + column[:, None]) / 2.0
    return np.stack([image, 255.0 - image]).astype(np.uint8)
