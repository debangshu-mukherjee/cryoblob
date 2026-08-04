# Changelog

All notable changes to **cryoblob** are documented here.

## [2026.7.0]

### Fixed
- **Package now runs on modern JAX.** The previous release did not execute
  under jax 0.4.38 / jaxtyping 0.2.38 (both within the declared support range).
  Fixed dynamic shapes under `@jax.jit` (`image_resizer`, `apply_gaussian_blur`,
  `resize_x`, `gaussian_kernel`, `histogram`/`equalize_hist`, `wiener`),
  inconsistent type annotations, and an invalid `lax.cond` in `make_mrc_image`.
- **Laplacian-of-Gaussian replaced with an analytic FFT LoG.** The former fixed
  202x202 spatial kernel was memory-heavy and truncated; the FFT implementation
  removes the memory ceiling and runs end-to-end at 128x128 through 4096x4096.
- **`blob_list_log` blob-size scaling.** Reported blob radii were `downscale`x
  too small: the position columns were scaled by `downscale` but the size column
  was not. Detection centers, F1, and localization are unaffected; this corrects
  the physical radius/size output.
- Connected-component labeling and centroiding now use `scipy.ndimage`
  (host-side) for correctness and stability.

### Added
- **`blob_list_log_watershed`** - LoG detection followed by a distance-transform
  watershed pass that recovers touching/overlapping particles that plain LoG
  merges. Keeps all LoG detections and adds only new watershed detections.
- **`plot_mrc(..., blobs=...)`** - overlay detected blobs on the image, drawn as
  circles at the detected positions and radii. Physical blob coordinates are
  converted back to pixels using the image voxel size (verified to sub-pixel
  accuracy against ground truth).

### Changed
- Added `scipy>=1.10.0` and `scikit-image>=0.22.0` to dependencies.
- Documentation: corrected the supported Python version (3.12+) and the
  validation stack (beartype + jaxtyping, not Pydantic).

## [2025.8.2]
- Prior released version (see git history).
