"""GPU-accelerated CLAHE for TensorFlow.

Authors: Bahador Mirzazadeh, Atefe Rostami

Quick start::

    import numpy as np
    import gpu_clahe

    images = np.random.randint(0, 256, (1000, 512, 512), dtype=np.uint8)
    enhanced = gpu_clahe.convert_clahe(images)

See :mod:`gpu_clahe.core` for the parameter conventions - in particular that
``clip_limit`` is a fraction of the pixels in a tile, not OpenCV's ``clipLimit``.
"""

from .api import convert_clahe
from .benchmark import benchmark_opencv, benchmark_performance, environment
from .config import CLAHEConfig, gpu_driver_version, total_gpu_memory_mb
from .core import clahe_gpu, clahe_gpu_nojit, setup_gpu
from .utils import get_gpu_info, require_valid_input, validate_input
from .version import __author__, __email__, __version__

__all__ = [
    "CLAHEConfig",
    "__author__",
    "__email__",
    "__version__",
    "benchmark_opencv",
    "benchmark_performance",
    "clahe_gpu",
    "clahe_gpu_nojit",
    "convert_clahe",
    "environment",
    "get_gpu_info",
    "gpu_driver_version",
    "require_valid_input",
    "setup_gpu",
    "total_gpu_memory_mb",
    "validate_input",
]
