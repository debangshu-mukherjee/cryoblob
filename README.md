[![PyPI Downloads](https://static.pepy.tech/badge/cryoblob)](https://pepy.tech/projects/cryoblob)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/cryoblob.svg)](https://badge.fury.io/py/cryoblob)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15548975.svg)](https://doi.org/10.5281/zenodo.15548975)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/debangshu-mukherjee/cryoblob/workflows/Tests/badge.svg)](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/cryoblob/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/cryoblob)
[![Documentation](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/docs.yml/badge.svg)](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/docs.yml)
[![Documentation Status](https://readthedocs.org/projects/cryoblob/badge/?version=latest)](https://cryoblob.readthedocs.io/en/latest/?badge=latest)

# cryoblob

**cryoblob** is a JAX-based, JIT-compiled, scalable package for reference-free detection of amorphous blobs in low-SNR cryo-EM images. It provides Laplacian-of-Gaussian blob detection for compact/circular particles, with an optional LoG+watershed mode that recovers touching or overlapping particles. (Experimental ridge/Hessian methods for elongated objects are included but are **not** reliable on low-SNR data — see [Scope and limitations](#scope-and-limitations).)

## Features

* **JAX-powered**: Leverages JAX for high-performance computing with automatic differentiation
* **GPU acceleration**: Can utilize both CPUs and GPUs for processing
* **Blob detection**:
  * **Laplacian-of-Gaussian** (`blob_list_log`): the validated, recommended detector for compact/circular particles in low-SNR data
  * **LoG + watershed** (`blob_list_log_watershed`, new in 2026.7.0): recovers touching/overlapping particles that plain LoG merges, at essentially unchanged precision
  * **Experimental** ridge/Hessian responses (`enhanced_blob_detection`) for elongated objects — included but *not* reliable under realistic low-SNR conditions (see Scope and limitations)
* **Differentiable response**: the LoG response is differentiable w.r.t. the image and scale σ (for gradient-based tuning of preprocessing/scale); particle selection itself is discrete and not differentiable
* **Adaptive filtering**: Includes adaptive Wiener filtering and thresholding
* **Batch processing**: Memory-optimized batch processing for large datasets
* **Validation**: Runtime type/shape validation using beartype and jaxtyping

## Scope and limitations

cryoblob is a reference-free, GPU-accelerated Laplacian-of-Gaussian **blob** detector for
low-SNR cryo-EM / materials micrographs — no labels, template, or known target size required.

* **Compact/round particles** are its validated use case (high recall and good localization).
* **Overlapping particles** are the principal failure mode; the optional
  `blob_list_log_watershed` mode partially recovers them (higher recall at similar precision),
  but dense clusters remain hard.
* **Solid elongated / filamentous features are out of scope.** LoG is a compact-blob detector,
  and the Hessian "ridge" response is non-selective under realistic noise; the ridge/elongated
  paths are experimental and did not yield reliable detection in low-SNR validation.
* cryoblob belongs to the same LoG family as classical reference-free pickers; its distinguishing
  features are the GPU/JAX implementation and a differentiable response — not higher accuracy
  than existing pickers.

## Installation

```bash
pip install cryoblob
```

## Quick Start

### Basic Blob Detection

```python
import cryoblob as cb

# Load an MRC file
mrc_image = cb.load_mrc("your_file.mrc")

# Traditional circular blob detection
blobs = cb.blob_list_log(mrc_image)

# Process a folder of images
results = cb.folder_blobs("path/to/folder/")

# Plot results (overlay the detected blobs on the image)
cb.plot_mrc(mrc_image, blobs=blobs)
```

### Overlapping particles (LoG + watershed)

```python
# Recovers touching/overlapping particles that plain LoG merges
blobs = cb.blob_list_log_watershed(mrc_image, min_blob_size=8, max_blob_size=110)
cb.plot_mrc(mrc_image, blobs=blobs)
```

### Experimental: multi-method detection

> **Experimental.** `enhanced_blob_detection` exposes ridge/Hessian and watershed
> responses for elongated and mixed morphologies. The ridge/elongated path is **not
> reliable under realistic low-SNR conditions** (see Scope and limitations); prefer
> `blob_list_log` / `blob_list_log_watershed` for compact and overlapping particles.

```python
from cryoblob.valid import (create_overlapping_blobs_pipeline,
                            create_comprehensive_pipeline)

# Overlapping circular structures (experimental multi-method path)
config = create_overlapping_blobs_pipeline()
circular, _, separated_blobs = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())

# All methods at once (experimental)
config = create_comprehensive_pipeline()
all_results = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())
```

## Detection Methods

| Blob type | Method | Best for | Key function |
|-----------|--------|----------|--------------|
| Circular / compact | LoG | Standard round particles (recommended) | `blob_list_log()` |
| Overlapping | LoG + watershed | Touching/overlapping particles | `blob_list_log_watershed()` |
| Elongated *(experimental)* | Ridge / Hessian | Not reliable under low SNR — see Scope and limitations | `enhanced_blob_detection()` |

## Package Structure

The cryoblob package is organized into the following modules:

* **adapt**: Adaptive image processing with gradient descent optimization
* **blobs**: Core blob detection algorithms and preprocessing  
* **files**: File I/O operations and batch processing
* **image**: Basic image processing functions (filtering, resizing, etc.)
* **multi**: Additional (experimental) multi-method detectors — Hessian/ridge and watershed variants
* **plots**: Visualization functions for MRC images and results
* **types**: Type definitions and PyTree structures
* **valid**: Parameter validation and presets (beartype + jaxtyping)

## Use Cases

**Standard cryo-EM particles (recommended)**
```python
# Reference-free LoG blob detection
blobs = cb.blob_list_log(mrc_image, min_blob_size=5, max_blob_size=20)
```

**Overlapping or touching particles**
```python
# LoG + watershed recovers particles that plain LoG merges
blobs = cb.blob_list_log_watershed(mrc_image, min_blob_size=5, max_blob_size=20)
```

**Elongated / filamentous structures (experimental — not reliable under low SNR)**
```python
# LoG is a compact-blob detector; the ridge/elongated path is experimental and
# was non-selective in low-SNR validation. See Scope and limitations.
_, elongated, _ = cb.enhanced_blob_detection(
    mrc_image, use_ridge_detection=True, use_watershed=False
)
```

## Performance

* **Memory Efficient**: Automatic batch size optimization and memory management
* **Scalable**: Multi-device and multi-host processing support
* **Fast**: JIT compilation and GPU acceleration where available
* **Flexible**: Selective method usage to optimize speed vs. comprehensiveness

## Package Organization
* The **codes** are located in `/src/cryoblob/`
* The **notebooks** are located in `/tutorials/`

## Documentation

For detailed API documentation and tutorials, visit: [https://cryoblob.readthedocs.io](https://cryoblob.readthedocs.io)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Debangshu Mukherjee (mukherjeed@ornl.gov)
- Alexis N. Williams (williamsan@ornl.gov)