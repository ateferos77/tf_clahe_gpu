import numpy as np
import tensorflow as tf
import time
from typing import Union, Tuple, Dict, Any


def validate_input(images: Union[np.ndarray, tf.Tensor]) -> Tuple[bool, str]:
    """
    Validate input images for CLAHE processing

    Returns:
        (is_valid, error_message)
    """
    if isinstance(images, np.ndarray):
        if images.ndim not in [3, 4]:
            return False, "Images must be 3D (batch, h, w) or 4D (batch, h, w, channels)"
        if images.dtype not in [np.uint8, np.float32, np.float64]:
            return False, "Images must be uint8, float32, or float64"
        if len(images) == 0:
            return False, "Empty image array"
    elif isinstance(images, tf.Tensor):
        if len(images.shape) not in [3, 4]:
            return False, "Tensor must be 3D or 4D"
    else:
        return False, "Input must be numpy array or tensorflow tensor"

    return True, ""


def benchmark_performance(
        image_shape: Tuple[int, int, int],
        num_runs: int = 5,
        batch_sizes: list = None
) -> Dict[str, Any]:
    """
    Benchmark CLAHE performance across different configurations

    Args:
        image_shape: Shape of test images (batch, h, w)
        num_runs: Number of benchmark runs
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128]

    results = {
        'image_shape': image_shape,
        'num_runs': num_runs,
        'batch_results': []
    }

    # Generate test data
    test_images = np.random.randint(0, 256, image_shape, dtype=np.uint8)

    from .core import batch_clahe_gpu, setup_gpu
    setup_gpu()

    for batch_size in batch_sizes:
        if batch_size > len(test_images):
            continue

        times = []
        for run in range(num_runs):
            # Warm up
            if run == 0:
                _ = batch_clahe_gpu(
                    tf.constant(test_images[:min(4, batch_size)], dtype=tf.uint8)
                )

            start_time = time.time()

            # Process in batches
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i + batch_size]
                _ = batch_clahe_gpu(tf.constant(batch, dtype=tf.uint8))

            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times[1:])  # Skip first run (warm-up)
        images_per_sec = len(test_images) / avg_time

        results['batch_results'].append({
            'batch_size': batch_size,
            'avg_time': avg_time,
            'images_per_sec': images_per_sec,
            'std_time': np.std(times[1:])
        })

    return results


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_info = {
            'num_gpus': len(gpus),
            'gpu_names': [gpu.name for gpu in gpus],
            'tensorflow_version': tf.__version__,
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_available': tf.test.is_gpu_available()
        }
        return gpu_info
    except Exception as e:
        return {'error': str(e)}
