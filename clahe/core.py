import tensorflow as tf
import numpy as np
from typing import Union, Literal
from numpy.typing import NDArray


@tf.function(jit_compile=True)
def clahe_gpu(
        images: tf.Tensor,
        tile_size: int = 32,
        clip_limit: float = 0.035,
        dtype: tf.DType = tf.uint8
) -> tf.Tensor:
    """
    GPU-optimized CLAHE - For both 3D and 4D inputs

    Args:
        images: Tensor of shape (batch, h, w) or (batch, h, w, 1)
        tile_size: Size of each tile for CLAHE
        clip_limit: Clipping limit for histogram equalization
        dtype: Output data type

    Returns:
        Processed images with same shape as input
    """

    # Handle input shapes properly
    input_rank = len(images.shape)

    # Convert to 3D if needed
    if input_rank == 4:
        images_3d = tf.squeeze(images, axis=-1)
        return_4d = True
    else:
        images_3d = images
        return_4d = False

    batch_size = tf.shape(images_3d)[0]
    h = tf.shape(images_3d)[1]
    w = tf.shape(images_3d)[2]

    # Pre-cast all constants to avoid repeated casting
    images_f32 = tf.cast(images_3d, tf.float32)
    tile_size_f32 = tf.cast(tile_size, tf.float32)
    tile_size_i32 = tf.cast(tile_size, tf.int32)
    clip_limit_f32 = tf.cast(clip_limit, tf.float32)
    h_f32 = tf.cast(h, tf.float32)
    w_f32 = tf.cast(w, tf.float32)

    # Compute tile dimensions using only GPU ops
    n_tiles_y = tf.cast(tf.math.ceil(h_f32 / tile_size_f32), tf.int32)
    n_tiles_x = tf.cast(tf.math.ceil(w_f32 / tile_size_f32), tf.int32)
    n_tiles_total = n_tiles_y * n_tiles_x

    # Padding computation
    pad_h = n_tiles_y * tile_size_i32 - h
    pad_w = n_tiles_x * tile_size_i32 - w

    # Symmetric padding for better edge handling
    images_padded = tf.pad(images_f32, [[0, 0], [0, pad_h], [0, pad_w]], mode='SYMMETRIC')
    padded_h = h + pad_h
    padded_w = w + pad_w

    # Ultra-efficient tile extraction using pure reshaping
    tile_pixels = tile_size_i32 * tile_size_i32

    # Reshape for tile extraction: (batch, n_tiles_y, tile_size, n_tiles_x, tile_size)
    reshaped = tf.reshape(images_padded, [batch_size, n_tiles_y, tile_size_i32, n_tiles_x, tile_size_i32])

    # Transpose to group tile pixels: (batch, n_tiles_y, n_tiles_x, tile_size, tile_size)
    tiles_5d = tf.transpose(reshaped, [0, 1, 3, 2, 4])

    # Final reshape to: (batch, n_tiles_total, tile_pixels)
    tiles = tf.reshape(tiles_5d, [batch_size, n_tiles_total, tile_pixels])

    # Convert to int32 for histogram operations with clamping
    tiles_int = tf.cast(tf.clip_by_value(tiles, 0.0, 255.0), tf.int32)

    # Ultra-fast vectorized histogram using broadcasting
    # Expand tiles: (batch, n_tiles, pixels, 1)
    tiles_expanded = tf.expand_dims(tiles_int, axis=-1)

    # Create bin values: (1, 1, 1, 256)
    bins = tf.reshape(tf.range(256, dtype=tf.int32), [1, 1, 1, 256])

    # Compute all histograms simultaneously: (batch, n_tiles, 256)
    histograms = tf.reduce_sum(
        tf.cast(tf.equal(tiles_expanded, bins), tf.int32),
        axis=2
    )

    # CLAHE processing - all vectorized
    clip_value = tf.cast(tf.cast(tile_pixels, tf.float32) * clip_limit_f32, tf.int32)

    # Clip histograms and redistribute excess
    excess = tf.maximum(histograms - clip_value, 0)
    clipped = tf.minimum(histograms, clip_value)
    redistribute = tf.reduce_sum(excess, axis=2, keepdims=True) // 256
    final_hist = clipped + redistribute

    # Compute CDFs for all tiles
    cdfs = tf.cumsum(final_hist, axis=2)

    # Normalize CDFs to create lookup tables
    cdf_min = cdfs[:, :, 0:1]
    cdf_max = cdfs[:, :, -1:]
    cdf_range = tf.cast(cdf_max - cdf_min, tf.float32)

    # Create normalized LUTs
    norm_factor = 255.0 / tf.maximum(cdf_range, 1.0)
    luts = tf.cast(
        (tf.cast(cdfs - cdf_min, tf.float32) * norm_factor),
        tf.int32
    )

    # Bilinear interpolation setup - vectorized coordinate computation
    y_coords = tf.cast(tf.range(padded_h), tf.float32) / tile_size_f32 - 0.5
    x_coords = tf.cast(tf.range(padded_w), tf.float32) / tile_size_f32 - 0.5

    # Create meshgrid
    yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')

    # Compute interpolation coordinates
    fy0 = tf.floor(yy)
    fx0 = tf.floor(xx)
    wy = yy - fy0
    wx = xx - fx0

    # Clamp tile indices
    fy0_i = tf.clip_by_value(tf.cast(fy0, tf.int32), 0, n_tiles_y - 1)
    fx0_i = tf.clip_by_value(tf.cast(fx0, tf.int32), 0, n_tiles_x - 1)
    fy1_i = tf.clip_by_value(fy0_i + 1, 0, n_tiles_y - 1)
    fx1_i = tf.clip_by_value(fx0_i + 1, 0, n_tiles_x - 1)

    # Compute linear tile indices for the four corners
    idx00 = fy0_i * n_tiles_x + fx0_i
    idx01 = fy0_i * n_tiles_x + fx1_i
    idx10 = fy1_i * n_tiles_x + fx0_i
    idx11 = fy1_i * n_tiles_x + fx1_i

    # Get pixel values
    pixel_vals = tf.cast(tf.clip_by_value(images_padded, 0.0, 255.0), tf.int32)

    # Optimized LUT application using advanced indexing
    def apply_luts_vectorized(
            luts_batch: tf.Tensor,
            tile_indices: tf.Tensor,
            pixel_values: tf.Tensor
    ) -> tf.Tensor:
        """Ultra-fast LUT application using vectorized operations"""

        # Create batch dimension for tile indices
        batch_range = tf.range(batch_size, dtype=tf.int32)
        batch_indices = tf.reshape(batch_range, [-1, 1, 1])
        batch_indices = tf.broadcast_to(batch_indices, [batch_size, padded_h, padded_w])

        # Expand tile indices to batch dimension
        tile_indices_batched = tf.expand_dims(tile_indices, 0)
        tile_indices_batched = tf.broadcast_to(tile_indices_batched, [batch_size, padded_h, padded_w])

        # Create gather indices for LUT selection: (batch, h, w, 2)
        lut_gather_idx = tf.stack([batch_indices, tile_indices_batched], axis=-1)

        # Gather LUTs: (batch, h, w, 256)
        selected_luts = tf.gather_nd(luts_batch, lut_gather_idx)

        # Apply LUTs to pixel values using advanced indexing
        h_range = tf.range(padded_h, dtype=tf.int32)
        w_range = tf.range(padded_w, dtype=tf.int32)

        h_indices = tf.reshape(h_range, [1, -1, 1])
        w_indices = tf.reshape(w_range, [1, 1, -1])

        h_indices = tf.broadcast_to(h_indices, [batch_size, padded_h, padded_w])
        w_indices = tf.broadcast_to(w_indices, [batch_size, padded_h, padded_w])

        # Final gather indices: (batch, h, w, 4)
        pixel_gather_idx = tf.stack([batch_indices, h_indices, w_indices, pixel_values], axis=-1)

        # Apply LUTs
        result = tf.gather_nd(selected_luts, pixel_gather_idx)
        return tf.cast(result, tf.float32)

    # Apply LUTs for all four interpolation corners
    val00 = apply_luts_vectorized(luts, idx00, pixel_vals)
    val01 = apply_luts_vectorized(luts, idx01, pixel_vals)
    val10 = apply_luts_vectorized(luts, idx10, pixel_vals)
    val11 = apply_luts_vectorized(luts, idx11, pixel_vals)

    # Bilinear interpolation
    val0 = val00 * (1.0 - wx) + val01 * wx
    val1 = val10 * (1.0 - wx) + val11 * wx
    interpolated = val0 * (1.0 - wy) + val1 * wy

    # Crop to original size
    result = interpolated[:, :h, :w]

    # Final type conversion
    result_clamped = tf.clip_by_value(result, 0.0, 255.0)
    result_typed = tf.cast(result_clamped, dtype)

    # Return in correct shape
    if return_4d:
        result_typed = tf.expand_dims(result_typed, axis=-1)

    return result_typed


@tf.function(jit_compile=True)
def clahe_gpu_wrapper(
        batch_tensor: tf.Tensor,
        tile_size: int,
        clip_limit: float
) -> tf.Tensor:
    """JIT-compiled wrapper function"""
    return clahe_gpu(batch_tensor, tile_size=tile_size, clip_limit=clip_limit, dtype=tf.uint8)


# Type aliases for better readability
ImageArray = Union[NDArray[np.uint8], tf.Tensor]
OutputType = Literal['numpy', 'tensor']
InputType = Literal['numpy', 'tensor']


def setup_gpu() -> None:
    """Safe GPU setup that works even after TensorFlow is initialized"""
    try:
        tf.config.optimizer.set_jit(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
    except Exception as e:
        print(f"GPU setup failed: {e}")


def _convert_output_format(
        results: Union[NDArray[np.uint8], tf.Tensor],
        return_tensor: bool = False
) -> Union[NDArray[np.uint8], tf.Tensor]:
    """Convert output to desired format"""
    if return_tensor:
        if isinstance(results, np.ndarray):
            return tf.convert_to_tensor(results, dtype=tf.uint8)
        return results
    else:
        if isinstance(results, tf.Tensor):
            return results.numpy()
        return results


def _convert_with_pipeline_hybrid(
        images: ImageArray,
        batch_size: int,
        tile_size: int,
        clip_limit: float,
        return_tensor: bool
) -> Union[NDArray[np.uint8], tf.Tensor]:
    """Enhanced pipeline with hybrid input/output support"""

    # Pre-allocate output array (always numpy for intermediate processing)
    if isinstance(images, tf.Tensor):
        output_shape = images.shape
        np_images = images.numpy()
    else:
        output_shape = images.shape
        np_images = images

    results = np.empty(output_shape, dtype=np.uint8)

    # Enhanced pipeline with more aggressive prefetching
    dataset = tf.data.Dataset.from_tensor_slices(np_images)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(3)  # More aggressive prefetching

    # Add memory cleanup hints
    processed_count = 0

    for batch_idx, batch in enumerate(dataset):
        # Process batch on GPU
        with tf.device('/GPU:0'):  # Explicit GPU placement
            processed_batch = clahe_gpu_wrapper(batch, tile_size, clip_limit)

        # Copy results and immediate cleanup
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(np_images))
        actual_batch_size = end_idx - start_idx

        results[start_idx:end_idx] = processed_batch[:actual_batch_size].numpy()

        # Explicit memory cleanup for large batches
        if batch_size > 64:
            del processed_batch

        processed_count += actual_batch_size

    # Convert to desired output format
    return _convert_output_format(results, return_tensor)


def _convert_with_batching_hybrid(
        images: ImageArray,
        batch_size: int,
        tile_size: int,
        clip_limit: float,
        return_tensor: bool = False
) -> Union[NDArray[np.uint8], tf.Tensor]:
    """Optimized batching with hybrid input/output support"""

    # Convert input to numpy for processing
    if isinstance(images, tf.Tensor):
        np_images = images.numpy()
        output_shape = images.shape
    else:
        np_images = images
        output_shape = images.shape

    # Pre-allocate output array
    results = np.empty(output_shape, dtype=np.uint8)

    for i in range(0, len(np_images), batch_size):
        end_idx = min(i + batch_size, len(np_images))
        batch = np_images[i:end_idx]

        # Use tf.convert_to_tensor for processing
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.uint8)

        # Process batch
        processed = clahe_gpu_wrapper(batch_tensor, tile_size, clip_limit)

        # Direct assignment to pre-allocated array
        results[i:end_idx] = processed.numpy()

    # Convert to desired output format
    return _convert_output_format(results, return_tensor)


def convert_clahe(
        images: ImageArray,
        batch_size: int = 128,
        tile_size: int = 32,
        clip_limit: float = 0.035,
        use_pipeline: bool = False,
        return_tensor: bool = False
) -> Union[NDArray[np.uint8], tf.Tensor]:
    """
    Ultra-optimized CLAHE conversion with hybrid input/output support

    Args:
        images: Input images (numpy.ndarray or tf.Tensor)
        batch_size: Batch size for processing
        tile_size: CLAHE tile size
        clip_limit: CLAHE clip limit
        use_pipeline: Whether to use tf.data pipeline (recommended for large datasets)
        return_tensor: If True, return tf.Tensor; if False, return numpy.ndarray

    Returns:
        Processed images in requested format (numpy array or tf.Tensor)
    """

    setup_gpu()

    # Get length based on input type
    if isinstance(images, tf.Tensor):
        total_images = tf.shape(images)[0].numpy()
    else:
        total_images = len(images)

    if use_pipeline and total_images > 1000:  # Use pipeline for large datasets
        result = _convert_with_pipeline_hybrid(images, batch_size, tile_size, clip_limit, return_tensor)
    else:
        result = _convert_with_batching_hybrid(images, batch_size, tile_size, clip_limit, return_tensor)
    return result
