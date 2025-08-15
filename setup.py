from setuptools import setup, find_packages
import os

# Read version
version_file = os.path.join(os.path.dirname(__file__), 'gpu_clahe', 'version.py')
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpu-clahe",
    version=version_dict['__version__'],
    author=version_dict['__author__'],
    author_email=version_dict['__email__'],
    description=version_dict['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Baha2rM98/gpu-clahe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "benchmark": [
            "opencv-python",
            "matplotlib",
            "seaborn",
            "memory-profiler",
        ]
    },
    keywords="gpu, clahe, image-processing, tensorflow, computer-vision, medical-imaging",
    project_urls={
        "Bug Reports": "https://github.com/Baha2rM98/gpu-clahe/issues",
        "Source": "https://github.com/Baha2rM98/gpu-clahe",
        "Documentation": "https://gpu-clahe.readthedocs.io/",
    },
)
