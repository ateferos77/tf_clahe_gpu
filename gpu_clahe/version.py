"""Single source of truth for package metadata.

``pyproject.toml`` reads ``__version__`` from here via setuptools' dynamic
version support, so this file must stay importable without any third-party
dependency (in particular, it must not import TensorFlow).
"""

__version__ = "2.0.1"
__author__ = "Bahador Mirzazadeh, Atefe Rostami"
__email__ = "baha2r.mirzazadeh98@gmail.com, ateferos77@gmail.com"
__description__ = "GPU-accelerated CLAHE implementation for TensorFlow."
