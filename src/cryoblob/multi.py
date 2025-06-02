"""
Module: multi
---------------------------

Multi-method blob detection for elongated objects and overlapping blobs.
Extends the base blob detection with ridge detection for elongated objects
and watershed segmentation for overlapping blobs.

Functions
---------
- `hessian_matrix_2d`:
    Compute Hessian matrix components for 2D image at given scale.
- `determinant_of_hessian`:
    Compute Determinant of Hessian for blob detection with better boundary detection.
- `ridge_detection`:
    Detect elongated objects using ridge detection with eigenvalue analysis.
- `multi_scale_ridge_detector`:
    Multi-scale ridge detection for various elongated object sizes.
- `distance_transform_euclidean`:
    Compute Euclidean distance transform for watershed marker generation.
- `watershed_segmentation`:
    Segment overlapping blobs using marker-based watershed algorithm.
- `adaptive_marker_generation`:
    Generate watershed markers using distance transform and local maxima.
- `hessian_blob_detection`:
    Detect blobs using Determinant of Hessian for better boundary detection.
- `enhanced_blob_detection`:
    Combined approach using LoG, ridge detection, and watershed segmentation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, jaxtyped

from cryoblob.blobs import center_of_mass_3d, find_connected_components
from cryoblob.image import apply_gaussian_blur, image_resizer
from cryoblob.types import MRC_Image, scalar_float, scalar_int, scalar_num

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def hessian_matrix_2d(
    image: Float[Array, "h w"], sigma: Optional[scalar_float] = 1.0
) -> Tuple[Float[Array, "h w"], Float[Array, "h w"], Float[Array, "h w"]]:
    """
    Description
    -----------
    Compute the Hessian matrix components for 2D image at given scale.
    Uses Gaussian derivatives to compute second-order partial derivatives.

    Parameters
    ----------
    - `image` (Float[Array, "h w"]):
        Input image
    - `sigma` (scalar_float, optional):
        Scale parameter for Gaussian derivatives. Default is 1.0

    Returns
    -------
    - `hxx` (Float[Array, "h w"]):
        Second derivative in x direction (horizontal)
    - `hxy` (Float[Array, "h w"]):
        Mixed second derivative (cross-correlation)
    - `hyy` (Float[Array, "h w"]):
        Second derivative in y direction (vertical)

    Flow
    ----
    - Apply Gaussian smoothing at specified scale
    - Compute first derivatives using Sobel operators
    - Compute second derivatives from first derivatives
    """
    blurred_image: Float[Array, "h w"] = apply_gaussian_blur(image, sigma=sigma)

    sobel_x: Float[Array, "3 3"] = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y: Float[Array, "3 3"] = sobel_x.T

    dx: Float[Array, "h w"] = jax.scipy.signal.convolve2d(
        blurred_image, sobel_x, mode="same"
    )
    dy: Float[Array, "h w"] = jax.scipy.signal.convolve2d(
        blurred_image, sobel_y, mode="same"
    )

    hxx: Float[Array, "h w"] = jax.scipy.signal.convolve2d(dx, sobel_x, mode="same")
    hxy: Float[Array, "h w"] = jax.scipy.signal.convolve2d(dx, sobel_y, mode="same")
    hyy: Float[Array, "h w"] = jax.scipy.signal.convolve2d(dy, sobel_y, mode="same")

    return hxx, hxy, hyy


@jaxtyped(typechecker=beartype)
def determinant_of_hessian(
    image: Float[Array, "h w"], sigma: Optional[scalar_float] = 1.0
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Compute Determinant of Hessian for blob detection with better boundary detection.
    DoH provides superior boundary localization compared to LoG for irregular shapes.

    Parameters
    ----------
    - `image` (Float[Array, "h w"]):
        Input image
    - `sigma` (scalar_float, optional):
        Scale parameter for Gaussian derivatives. Default is 1.0

    Returns
    -------
    - `normalized_det` (Float[Array, "h w"]):
        Scale-normalized determinant of Hessian response

    Flow
    ----
    - Compute Hessian matrix components
    - Calculate determinant: det(H) = Hxx * Hyy - Hxy^2
    - Apply scale normalization: sigma^4 * det(H)
    """
    hxx: Float[Array, "h w"]
    hxy: Float[Array, "h w"]
    hyy: Float[Array, "h w"]
    hxx, hxy, hyy = hessian_matrix_2d(image, sigma)

    det_hessian: Float[Array, "h w"] = hxx * hyy - hxy * hxy
    normalized_det: Float[Array, "h w"] = sigma**4 * det_hessian

    return normalized_det


@jaxtyped(typechecker=beartype)
def ridge_detection(
    image: Float[Array, "h w"],
    sigma: Optional[scalar_float] = 1.0,
    threshold: Optional[scalar_float] = 0.01,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Detect ridges (elongated objects) using eigenvalues of Hessian matrix.
    Ridges correspond to structures with one large negative eigenvalue.

    Parameters
    ----------
    - `image` (Float[Array, "h w"]):
        Input image
    - `sigma` (scalar_float, optional):
        Scale parameter for Gaussian derivatives. Default is 1.0
    - `threshold` (scalar_float, optional):
        Ridge strength threshold. Default is 0.01

    Returns
    -------
    - `ridge_response` (Float[Array, "h w"]):
        Ridge detection response map

    Flow
    ----
    - Compute Hessian matrix components
    - Calculate eigenvalues using trace and determinant
    - Extract ridge strength as maximum absolute eigenvalue
    - Apply threshold to create ridge response map
    """
    hxx: Float[Array, "h w"]
    hxy: Float[Array, "h w"]
    hyy: Float[Array, "h w"]
    hxx, hxy, hyy = hessian_matrix_2d(image, sigma)

    trace: Float[Array, "h w"] = hxx + hyy
    det: Float[Array, "h w"] = hxx * hyy - hxy * hxy

    discriminant: Float[Array, "h w"] = trace**2 - 4 * det
    discriminant = jnp.maximum(discriminant, 0.0)

    sqrt_discriminant: Float[Array, "h w"] = jnp.sqrt(discriminant)
    eigenvalue1: Float[Array, "h w"] = (trace + sqrt_discriminant) / 2
    eigenvalue2: Float[Array, "h w"] = (trace - sqrt_discriminant) / 2

    ridge_strength: Float[Array, "h w"] = jnp.maximum(
        jnp.abs(eigenvalue1), jnp.abs(eigenvalue2)
    )

    ridge_response: Float[Array, "h w"] = jnp.where(
        ridge_strength > threshold, ridge_strength, 0.0
    )

    return ridge_response


@jaxtyped(typechecker=beartype)
def multi_scale_ridge_detector(
    image: Float[Array, "h w"],
    min_scale: Optional[scalar_float] = 1.0,
    max_scale: Optional[scalar_float] = 10.0,
    num_scales: Optional[scalar_int] = 10,
    threshold: Optional[scalar_float] = 0.01,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Multi-scale ridge detection for elongated objects of various sizes.
    Detects ridges across multiple scales and returns maximum response.

    Parameters
    ----------
    - `image` (Float[Array, "h w"]):
        Input image
    - `min_scale` (scalar_float, optional):
        Minimum scale parameter. Default is 1.0
    - `max_scale` (scalar_float, optional):
        Maximum scale parameter. Default is 10.0
    - `num_scales` (scalar_int, optional):
        Number of scales to test. Default is 10
    - `threshold` (scalar_float, optional):
        Ridge detection threshold. Default is 0.01

    Returns
    -------
    - `max_ridge_response` (Float[Array, "h w"]):
        Maximum ridge response across all scales

    Flow
    ----
    - Generate logarithmic scale sequence
    - Apply ridge detection at each scale
    - Take maximum response across scales
    """
    scales: Float[Array, "num_scales"] = jnp.linspace(min_scale, max_scale, num_scales)

    def compute_ridge_at_scale(scale: scalar_float) -> Float[Array, "h w"]:
        return ridge_detection(image, sigma=scale, threshold=threshold)

    ridge_responses: Float[Array, "num_scales h w"] = jax.vmap(compute_ridge_at_scale)(
        scales
    )

    max_ridge_response: Float[Array, "h w"] = jnp.max(ridge_responses, axis=0)

    return max_ridge_response


@jaxtyped(typechecker=beartype)
def distance_transform_euclidean(
    binary_image: Bool[Array, "h w"],
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Compute Euclidean distance transform for watershed marker generation.
    Calculates minimum distance from each background pixel to nearest foreground pixel.

    Parameters
    ----------
    - `binary_image` (Bool[Array, "h w"]):
        Binary image where True represents foreground objects

    Returns
    -------
    - `distance_map` (Float[Array, "h w"]):
        Euclidean distance transform map

    Flow
    ----
    - Extract coordinates of all foreground pixels
    - For each background pixel, compute minimum distance to foreground
    - Set foreground pixels to distance zero
    """
    h: scalar_int
    w: scalar_int
    h, w = binary_image.shape

    y_coords: Float[Array, "h w"]
    x_coords: Float[Array, "h w"]
    y_coords, x_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

    foreground_mask: Bool[Array, "h w"] = binary_image
    foreground_y: Float[Array, "n"] = y_coords[foreground_mask]
    foreground_x: Float[Array, "n"] = x_coords[foreground_mask]

    def compute_min_distance_to_foreground(
        y: scalar_int, x: scalar_int
    ) -> scalar_float:
        if len(foreground_y) == 0:
            return jnp.inf

        distances: Float[Array, "n"] = jnp.sqrt(
            (foreground_y - y) ** 2 + (foreground_x - x) ** 2
        )
        return jnp.min(distances)

    distance_map: Float[Array, "h w"] = jnp.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if not binary_image[i, j]:
                distance_map = distance_map.at[i, j].set(
                    compute_min_distance_to_foreground(i, j)
                )

    return distance_map


@jaxtyped(typechecker=beartype)
def adaptive_marker_generation(
    binary_image: Bool[Array, "h w"], min_distance: Optional[scalar_float] = 5.0
) -> Integer[Array, "h w"]:
    """
    Description
    -----------
    Generate watershed markers using distance transform and local maxima detection.
    Creates seed points for watershed segmentation of overlapping objects.

    Parameters
    ----------
    - `binary_image` (Bool[Array, "h w"]):
        Binary image of detected blob regions
    - `min_distance` (scalar_float, optional):
        Minimum distance threshold for marker placement. Default is 5.0

    Returns
    -------
    - `markers` (Integer[Array, "h w"]):
        Marker image with labeled seed regions for watershed

    Flow
    ----
    - Compute distance transform of binary image
    - Find local maxima in distance transform
    - Filter maxima by minimum distance threshold
    - Label connected components to create markers
    - Set background regions to -1, unlabeled foreground to 0
    """
    distance_map: Float[Array, "h w"] = distance_transform_euclidean(binary_image)

    h: scalar_int
    w: scalar_int
    h, w = distance_map.shape
    local_maxima: Bool[Array, "h w"] = jnp.zeros_like(binary_image)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center_value: scalar_float = distance_map[i, j]

            is_foreground: bool = binary_image[i, j]
            is_above_threshold: bool = center_value >= min_distance

            neighborhood: Float[Array, "3 3"] = distance_map[
                i - 1 : i + 2, j - 1 : j + 2
            ]
            is_local_maximum: bool = jnp.all(neighborhood <= center_value)

            local_maxima = local_maxima.at[i, j].set(
                is_foreground and is_above_threshold and is_local_maximum
            )

    labeled_markers: Integer[Array, "h w"]
    num_markers: scalar_int
    labeled_markers, num_markers = find_connected_components(local_maxima)

    markers: Integer[Array, "h w"] = jnp.where(
        binary_image & (labeled_markers == 0), 0, labeled_markers
    )
    markers = jnp.where(~binary_image, -1, markers)

    return markers


@jaxtyped(typechecker=beartype)
def watershed_segmentation(
    image: Float[Array, "h w"],
    markers: Integer[Array, "h w"],
    max_iterations: Optional[scalar_int] = 15,
) -> Integer[Array, "h w"]:
    """
    Description
    -----------
    Marker-based watershed segmentation for separating overlapping blobs.
    Simulates flooding from marked seed points to segment touching objects.

    Parameters
    ----------
    - `image` (Float[Array, "h w"]):
        Input image (typically gradient magnitude or distance transform)
    - `markers` (Integer[Array, "h w"]):
        Marker image with labeled seed points
    - `max_iterations` (scalar_int, optional):
        Maximum number of flooding iterations. Default is 15

    Returns
    -------
    - `segmented` (Integer[Array, "h w"]):
        Segmented image with watershed boundaries marked as -1

    Flow
    ----
    - Initialize segmentation with marker labels
    - Iteratively expand labeled regions to neighboring pixels
    - Handle conflicts at boundaries between different labels
    - Stop when no more pixels can be labeled or max iterations reached
    """
    h: scalar_int
    w: scalar_int
    h, w = image.shape

    segmented: Integer[Array, "h w"] = markers.copy()

    def flooding_step(
        carry: Integer[Array, "h w"], iteration: scalar_int
    ) -> Tuple[Integer[Array, "h w"], None]:
        current_seg: Integer[Array, "h w"] = carry
        updated_seg: Integer[Array, "h w"] = current_seg.copy()

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if current_seg[i, j] == 0:
                    neighbors: Integer[Array, "8"] = jnp.array(
                        [
                            current_seg[i - 1, j - 1],
                            current_seg[i - 1, j],
                            current_seg[i - 1, j + 1],
                            current_seg[i, j - 1],
                            current_seg[i, j + 1],
                            current_seg[i + 1, j - 1],
                            current_seg[i + 1, j],
                            current_seg[i + 1, j + 1],
                        ]
                    )

                    positive_neighbors: Integer[Array, "m"] = neighbors[neighbors > 0]

                    if len(positive_neighbors) > 0:
                        unique_labels: Integer[Array, "k"] = jnp.unique(
                            positive_neighbors
                        )

                        if len(unique_labels) == 1:
                            updated_seg = updated_seg.at[i, j].set(unique_labels[0])
                        else:
                            updated_seg = updated_seg.at[i, j].set(-1)

        return updated_seg, None

    final_seg: Integer[Array, "h w"]
    final_seg, _ = lax.scan(flooding_step, segmented, jnp.arange(max_iterations))

    return final_seg


@jaxtyped(typechecker=beartype)
def hessian_blob_detection(
    mrc_image: MRC_Image,
    min_blob_size: Optional[scalar_num] = 5,
    max_blob_size: Optional[scalar_num] = 20,
    blob_step: Optional[scalar_num] = 1,
    downscale: Optional[scalar_num] = 4,
    std_threshold: Optional[scalar_num] = 6,
) -> Float[Array, "n 3"]:
    """
    Description
    -----------
    Detect blobs using Determinant of Hessian for superior boundary detection.
    DoH provides better localization of blob boundaries compared to LoG method.

    Parameters
    ----------
    - `mrc_image` (MRC_Image):
        Input MRC image structure
    - `min_blob_size` (scalar_num, optional):
        Minimum blob size to detect. Default is 5
    - `max_blob_size` (scalar_num, optional):
        Maximum blob size to detect. Default is 20
    - `blob_step` (scalar_num, optional):
        Step size between consecutive scales. Default is 1
    - `downscale` (scalar_num, optional):
        Image downscaling factor for computational efficiency. Default is 4
    - `std_threshold` (scalar_num, optional):
        Detection threshold in standard deviations. Default is 6

    Returns
    -------
    - `scaled_coords` (Float[Array, "n 3"]):
        Detected blob coordinates and sizes [Y_nm, X_nm, Size_nm]

    Flow
    ----
    - Downscale image for computational efficiency
    - Apply DoH detection across multiple scales
    - Use 3D connected components for blob extraction
    - Scale coordinates back to original image dimensions
    """
    image: Float[Array, "H W"] = mrc_image.image_data.astype(jnp.float32)
    voxel_size: Float[Array, "3"] = mrc_image.voxel_size
    scales: Float[Array, "num_scales"] = jnp.arange(
        min_blob_size, max_blob_size, blob_step
    )

    scaled_image: Float[Array, "h w"] = image_resizer(image, downscale)

    def apply_hessian(img: Float[Array, "h w"], sigma: scalar_float):
        return determinant_of_hessian(img, sigma=sigma)

    hessian_responses: Float[Array, "num_scales h w"] = jax.vmap(
        apply_hessian, in_axes=(None, 0)
    )(scaled_image, scales)

    hessian_3d: Float[Array, "h w num_scales"] = hessian_responses.transpose(1, 2, 0)

    max_filtered: Float[Array, "h w num_scales"] = jax.lax.reduce_window(
        operand=hessian_3d,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(3, 3, 3),
        window_strides=(1, 1, 1),
        padding="SAME",
    )

    mean_val: scalar_float = jnp.mean(max_filtered)
    std_val: scalar_float = jnp.std(max_filtered)
    threshold: scalar_float = mean_val + std_threshold * std_val

    binary_detections: Bool[Array, "h w num_scales"] = max_filtered > threshold
    labels: Integer[Array, "h w num_scales"]
    num_labels: scalar_int
    labels, num_labels = find_connected_components(binary_detections)

    if num_labels == 0:
        return jnp.array([]).reshape(0, 3)

    coords: Float[Array, "n 3"] = center_of_mass_3d(hessian_3d, labels, num_labels)

    scaled_coords: Float[Array, "n 3"] = jnp.concatenate(
        [
            downscale * coords[:, :2] * voxel_size[1:][::-1],
            (coords[:, 2:] * blob_step + min_blob_size)[:, None]
            * jnp.sqrt(voxel_size[1] * voxel_size[2]),
        ],
        axis=1,
    )

    return scaled_coords


@jaxtyped(typechecker=beartype)
def enhanced_blob_detection(
    mrc_image: MRC_Image,
    min_blob_size: Optional[scalar_num] = 5,
    max_blob_size: Optional[scalar_num] = 20,
    blob_step: Optional[scalar_num] = 1,
    downscale: Optional[scalar_num] = 4,
    std_threshold: Optional[scalar_num] = 6,
    use_ridge_detection: Optional[bool] = True,
    use_watershed: Optional[bool] = True,
    ridge_threshold: Optional[scalar_float] = 0.01,
    min_marker_distance: Optional[scalar_float] = 5.0,
) -> Tuple[Float[Array, "n 3"], Float[Array, "m 3"], Float[Array, "k 3"]]:
    """
    Description
    -----------
    Enhanced multi-method blob detection combining Hessian blobs, ridge detection,
    and watershed segmentation for comprehensive analysis of circular blobs,
    elongated objects, and overlapping structures.

    Parameters
    ----------
    - `mrc_image` (MRC_Image):
        Input MRC image structure
    - `min_blob_size` (scalar_num, optional):
        Minimum blob size to detect. Default is 5
    - `max_blob_size` (scalar_num, optional):
        Maximum blob size to detect. Default is 20
    - `blob_step` (scalar_num, optional):
        Step size between consecutive scales. Default is 1
    - `downscale` (scalar_num, optional):
        Image downscaling factor for efficiency. Default is 4
    - `std_threshold` (scalar_num, optional):
        Detection threshold in standard deviations. Default is 6
    - `use_ridge_detection` (bool, optional):
        Enable ridge detection for elongated objects. Default is True
    - `use_watershed` (bool, optional):
        Enable watershed segmentation for overlapping blobs. Default is True
    - `ridge_threshold` (scalar_float, optional):
        Ridge detection sensitivity threshold. Default is 0.01
    - `min_marker_distance` (scalar_float, optional):
        Minimum distance between watershed markers. Default is 5.0

    Returns
    -------
    - `circular_blobs` (Float[Array, "n 3"]):
        Circular/irregular blob detections using Hessian method
    - `elongated_blobs` (Float[Array, "m 3"]):
        Elongated object detections using ridge detection
    - `watershed_blobs` (Float[Array, "k 3"]):
        Separated overlapping blob detections using watershed

    Flow
    ----
    - Apply Hessian-based blob detection for general blob shapes
    - Optionally apply ridge detection for elongated structures
    - Optionally apply watershed segmentation for overlapping blobs
    - Return separate detection results for analysis and combination
    """
    image: Float[Array, "H W"] = mrc_image.image_data.astype(jnp.float32)
    voxel_size: Float[Array, "3"] = mrc_image.voxel_size
    scaled_image: Float[Array, "h w"] = image_resizer(image, downscale)

    circular_blobs: Float[Array, "n 3"] = hessian_blob_detection(
        mrc_image, min_blob_size, max_blob_size, blob_step, downscale, std_threshold
    )

    elongated_blobs: Float[Array, "m 3"] = jnp.array([]).reshape(0, 3)
    if use_ridge_detection:
        ridge_response: Float[Array, "h w"] = multi_scale_ridge_detector(
            scaled_image,
            min_scale=min_blob_size / 2,
            max_scale=max_blob_size * 2,
            num_scales=15,
            threshold=ridge_threshold,
        )

        ridge_binary: Bool[Array, "h w"] = ridge_response > ridge_threshold
        ridge_labels: Integer[Array, "h w"]
        num_ridge_labels: scalar_int
        ridge_labels, num_ridge_labels = find_connected_components(ridge_binary)

        if num_ridge_labels > 0:
            ridge_coords_2d: Float[Array, "num_ridge_labels 2"] = center_of_mass_3d(
                ridge_response[:, :, None], ridge_labels[:, :, None], num_ridge_labels
            )[:, :2]

            ridge_sizes: Float[Array, "num_ridge_labels 1"] = (
                jnp.ones((num_ridge_labels, 1)) * (min_blob_size + max_blob_size) / 2
            )

            elongated_blobs = jnp.concatenate(
                [
                    downscale * ridge_coords_2d * voxel_size[1:][::-1],
                    ridge_sizes * jnp.sqrt(voxel_size[1] * voxel_size[2]),
                ],
                axis=1,
            )

    watershed_blobs: Float[Array, "k 3"] = jnp.array([]).reshape(0, 3)
    if use_watershed and len(circular_blobs) > 0:
        blob_binary: Bool[Array, "h w"] = jnp.zeros_like(scaled_image, dtype=bool)

        scaled_blob_coords: Float[Array, "n 2"] = jnp.column_stack(
            [
                circular_blobs[:, 0] / (downscale * voxel_size[1]),
                circular_blobs[:, 1] / (downscale * voxel_size[2]),
            ]
        )

        scaled_blob_radii: Float[Array, "n"] = circular_blobs[:, 2] / (
            2 * jnp.sqrt(voxel_size[1] * voxel_size[2])
        )

        h: scalar_int
        w: scalar_int
        h, w = scaled_image.shape
        y_coords: Float[Array, "h w"]
        x_coords: Float[Array, "h w"]
        y_coords, x_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

        for i in range(len(circular_blobs)):
            y_center: scalar_float = scaled_blob_coords[i, 0]
            x_center: scalar_float = scaled_blob_coords[i, 1]
            radius: scalar_float = scaled_blob_radii[i]

            distances: Float[Array, "h w"] = jnp.sqrt(
                (y_coords - y_center) ** 2 + (x_coords - x_center) ** 2
            )
            blob_binary = blob_binary | (distances <= radius)

        markers: Integer[Array, "h w"] = adaptive_marker_generation(
            blob_binary, min_distance=min_marker_distance
        )

        gradient_y: Float[Array, "h w"] = jnp.gradient(scaled_image, axis=0)
        gradient_x: Float[Array, "h w"] = jnp.gradient(scaled_image, axis=1)
        gradient_magnitude: Float[Array, "h w"] = jnp.sqrt(
            gradient_y**2 + gradient_x**2
        )

        watershed_labels: Integer[Array, "h w"] = watershed_segmentation(
            gradient_magnitude, markers, max_iterations=12
        )

        unique_labels: Integer[Array, "unique_count"] = jnp.unique(
            watershed_labels[watershed_labels > 0]
        )

        if len(unique_labels) > 0:
            watershed_coords: Float[Array, "unique_count 2"] = center_of_mass_3d(
                scaled_image[:, :, None],
                watershed_labels[:, :, None],
                len(unique_labels),
            )[:, :2]

            watershed_sizes: Float[Array, "unique_count 1"] = (
                jnp.ones((len(unique_labels), 1)) * (min_blob_size + max_blob_size) / 2
            )

            watershed_blobs = jnp.concatenate(
                [
                    downscale * watershed_coords * voxel_size[1:][::-1],
                    watershed_sizes * jnp.sqrt(voxel_size[1] * voxel_size[2]),
                ],
                axis=1,
            )

    return circular_blobs, elongated_blobs, watershed_blobs
