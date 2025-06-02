[![PyPI Downloads](https://static.pepy.tech/badge/cryoblob)](https://pepy.tech/projects/cryoblob)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/cryoblob.svg)](https://badge.fury.io/py/cryoblob)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15548975.svg)](https://doi.org/10.5281/zenodo.15548975)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/debangshu-mukherjee/cryoblob/workflows/Tests/badge.svg)](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/cryoblob/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/cryoblob)
[![Documentation](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/docs.yml/badge.svg)](https://github.com/debangshu-mukherjee/cryoblob/actions/workflows/docs.yml)
[![Documentation Status](https://readthedocs.org/projects/cryoblob/badge/?version=latest)](https://cryoblob.readthedocs.io/en/latest/?badge=latest)

# cryoblob

**cryoblob** is a JAX-based, JIT-compiled, scalable package for detection of amorphous blobs in low SNR cryo-EM images. It provides both traditional circular blob detection and advanced multi-method detection for complex morphologies including elongated objects and overlapping structures.

## Features

* **JAX-powered**: Leverages JAX for high-performance computing with automatic differentiation
* **GPU acceleration**: Can utilize both CPUs and GPUs for processing
* **Multi-method detection**: Advanced detection algorithms for diverse blob morphologies:
  * **Traditional LoG**: Excellent for circular blobs
  * **Ridge detection**: Specialized for elongated (pill-shaped) objects
  * **Watershed segmentation**: Separates overlapping circular structures
  * **Hessian-based detection**: Superior boundary localization
* **Adaptive filtering**: Includes adaptive Wiener filtering and thresholding
* **Batch processing**: Memory-optimized batch processing for large datasets
* **Validation**: Comprehensive parameter validation using Pydantic models

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

# Plot results
cb.plot_mrc(mrc_image)
```

### Enhanced Multi-Method Detection

```python
# For complex scenarios with multiple blob types
circular, elongated, overlapping = cb.enhanced_blob_detection(
    mrc_image,
    use_ridge_detection=True,    # Detect elongated objects
    use_watershed=True           # Separate overlapping blobs
)

print(f"Found {len(circular)} circular, {len(elongated)} elongated, "
      f"and {len(overlapping)} overlapping blobs")
```

### Specialized Detection

```python
from cryoblob.valid import (create_elongated_objects_pipeline, 
                           create_overlapping_blobs_pipeline,
                           create_comprehensive_pipeline)

# For elongated (pill-shaped) objects
config = create_elongated_objects_pipeline()
_, elongated_blobs, _ = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())

# For overlapping circular structures  
config = create_overlapping_blobs_pipeline()
circular, _, separated_blobs = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())

# For comprehensive analysis (all methods)
config = create_comprehensive_pipeline()
all_results = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())
```

## Detection Methods

| Blob Type | Method | Best For | Key Function |
|-----------|--------|----------|--------------|
| Circular | LoG | Standard round particles | `blob_list_log()` |
| Elongated | Ridge Detection | Pill-shaped, rod-like objects | `ridge_detection()` |
| Overlapping | Watershed | Touching circular structures | `watershed_segmentation()` |
| Mixed/Complex | Enhanced Detection | Multiple morphologies | `enhanced_blob_detection()` |

## Package Structure

The cryoblob package is organized into the following modules:

* **adapt**: Adaptive image processing with gradient descent optimization
* **blobs**: Core blob detection algorithms and preprocessing  
* **files**: File I/O operations and batch processing
* **image**: Basic image processing functions (filtering, resizing, etc.)
* **multi**: Multi-method detection for elongated objects and overlapping blobs
* **plots**: Visualization functions for MRC images and results
* **types**: Type definitions and PyTree structures
* **valid**: Parameter validation using Pydantic models

## Use Cases

**Standard Cryo-EM Particles**
```python
# Traditional circular blob detection
blobs = cb.blob_list_log(mrc_image, min_blob_size=5, max_blob_size=20)
```

**Elongated Biological Structures**
```python
# Detect pill-shaped, rod-like, or filamentous objects
_, elongated, _ = cb.enhanced_blob_detection(
    mrc_image, use_ridge_detection=True, use_watershed=False
)
```

**Overlapping or Touching Particles**
```python
# Separate overlapping circular structures
_, _, separated = cb.enhanced_blob_detection(
    mrc_image, use_ridge_detection=False, use_watershed=True
)
```

**Complex Heterogeneous Samples**
```python
# Comprehensive analysis for mixed morphologies
circular, elongated, overlapping = cb.enhanced_blob_detection(
    mrc_image, use_ridge_detection=True, use_watershed=True
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