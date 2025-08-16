import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from clahe.core import convert_clahe, setup_gpu


@dataclass
class CLAHEConfig:
    """Extended configuration for CLAHE processing."""
    tile_size: int = 32
    clip_limit: float = 0.035
    batch_size: int = 128
    dtype: tf.dtypes.DType = tf.uint8
    enable_xla: bool = True
    memory_growth: bool = True
    use_pipeline: bool = False
    return_tensor: bool = False

    def auto_batch_size(self, image_shape: Tuple[int, ...]) -> int:
        """Suggest a batch size based on GPU memory and image size."""

        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                info: Dict[str, int] = tf.config.experimental.get_memory_info('GPU:0')
                # available memory = total memory - current in use
                available_mb = (info['peak'] - info['current']) / (1024 ** 2)
            else:
                available_mb = 0
        except Exception:
            available_mb = 0

        # estimate per-image memory footprint (4 bytes per pixel)
        h, w = image_shape[-2], image_shape[-1]
        img_mb = (h * w * 4) / (1024 ** 2)
        if img_mb == 0:
            return self.batch_size

        # scale batch size conservatively
        estimate = int(max(8, min(128, available_mb / (img_mb * 15))))
        return estimate

    def _to_kwargs(self) -> Dict[str, Any]:
        """Convert config to keyword arguments for convert_clahe()."""
        return {
            'batch_size': self.batch_size,
            'tile_size': self.tile_size,
            'clip_limit': self.clip_limit,
            'use_pipeline': self.use_pipeline,
            'return_tensor': self.return_tensor,
        }

    def _setup_gpu(self) -> None:
        """Apply XLA and memory growth settings."""
        if self.enable_xla:
            tf.config.optimizer.set_jit(True)
        if self.memory_growth:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        # warm up TensorFlow to allocate memory
        setup_gpu()

    def apply(self, images):
        """Run CLAHE with this configuration on the provided images."""
        # update batch size if auto‑calculation is desired
        if self.batch_size <= 0:
            self.batch_size = self.auto_batch_size(images.shape)
        self._setup_gpu()
        return convert_clahe(images, **self._to_kwargs())
