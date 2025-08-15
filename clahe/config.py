import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CLAHEConfig:
    """Configuration for CLAHE processing"""

    tile_size: int = 32
    clip_limit: float = 0.035
    dtype: tf.DType = tf.uint8
    enable_xla: bool = True
    memory_growth: bool = True

    def auto_batch_size(self, image_shape: Tuple[int, ...]) -> int:
        """
        Automatically determine optimal batch size based on:
        - GPU memory
        - Image dimensions
        - Available VRAM
        """
        try:
            # Get GPU memory info
            gpu_memory = self._get_gpu_memory_mb()
            image_size_mb = self._estimate_image_memory(image_shape)

            # Conservative batch size calculation
            if gpu_memory > 20000:  # 20GB+
                return min(128, max(32, int(gpu_memory / (image_size_mb * 10))))
            elif gpu_memory > 10000:  # 10GB+
                return min(64, max(16, int(gpu_memory / (image_size_mb * 15))))
            else:  # <10GB
                return min(32, max(8, int(gpu_memory / (image_size_mb * 20))))

        except:
            return 32  # Safe default

    @staticmethod
    def _get_gpu_memory_mb() -> int:
        """Get available GPU memory in MB"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # This is a simplified estimation
                return 8000  # Default assumption
            return 0
        except:
            return 0

    @staticmethod
    def _estimate_image_memory(shape: Tuple[int, ...]) -> float:
        """Estimate memory usage per image in MB"""
        if len(shape) >= 3:
            pixels = shape[-2] * shape[-1]  # H * W
            return (pixels * 4) / (1024 * 1024)  # 4 bytes per pixel (float32)
        return 1.0
