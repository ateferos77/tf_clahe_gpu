"""
Example usage for tf_clahe_gpu.

This script loads images from a .npz file or generates dummy data if that file isn’t found,
runs CLAHE on the GPU, and saves the result. Run it directly with Python.
"""

from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from clahe import convert_clahe


def load_images(from_file_name):
    path = os.path.join(os.path.dirname(os.getcwd()), 'dataset', from_file_name)
    p = Path(path)
    if p.exists():
        data = np.load(p)

        def fix_shape(arr, name):
            # Accept (batch, h, w) or (batch, h, w, 1)
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim != 3:
                raise ValueError(f"{name} has invalid shape {arr.shape}, expected (batch, h, w) or (batch, h, w, 1)")
            return arr.astype(np.uint8)

        # train = fix_shape(data['train_images'], 'train_images')
        # test = fix_shape(data['test_images'], 'test_images')
        val = fix_shape(data['val_images'], 'val_images')

        # images = np.concatenate([train, test, val], axis=0)
        return val

    # fallback: create 10 random 512×512 images for testing
    print(f"{p} not found; generating random images.")
    return np.random.randint(0, 256, (10, 512, 512), dtype=np.uint8)


def plot_before_after(original, processed, title_left="Original", title_right="CLAHE"):
    # Ensure 2D for plotting
    if original.ndim == 3 and original.shape[-1] == 1:
        original = original[..., 0]
    if processed.ndim == 3 and processed.shape[-1] == 1:
        processed = processed[..., 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(title_left)
    axes[0].axis("off")

    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(title_right)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main(seed=None):
    images = load_images(from_file_name='chestmnist_224.npz')
    print(f"Loaded {len(images)} images of shape {images.shape[1:]}.")

    # Run CLAHE on GPU
    clahe_images = convert_clahe(images)

    # Pick a random sample (stable if seed provided)
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(images)))

    print(f"Showing sample index {idx}.")
    plot_before_after(images[idx], clahe_images[idx])


if __name__ == "__main__":
    # Set a number for reproducible randomness, or None for fresh chaos every time
    main(seed=None)
