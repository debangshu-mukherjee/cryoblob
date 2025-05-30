[![PyPI Downloads](https://static.pepy.tech/badge/cryoblob)](https://pepy.tech/projects/cryoblob)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/cryoblob.svg)](https://badge.fury.io/py/cryoblob)
[![Documentation Status](https://readthedocs.org/projects/cryoblob/badge/?version=latest)](https://cryoblob.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15548975.svg)](https://doi.org/10.5281/zenodo.15548975)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/cryoblob/branch/main/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/cryoblob)
[![Tests](https://github.com/debangshu-mukherjee/cryoblob/workflows/Tests/badge.svg)](https://github.com/debangshu-mukherjee/cryoblob/actions)

# cryoblob

**cryoblob** is a JAX-based, JIT-compiled, scalable package for detection of amorphous blobs in low SNR cryo-EM images.

## Features

* **JAX-powered**: Leverages JAX for high-performance computing with automatic differentiation
* **GPU acceleration**: Can utilize both CPUs and GPUs for processing
* **Adaptive filtering**: Includes adaptive Wiener filtering and thresholding
* **Blob detection**: Advanced blob detection using Laplacian of Gaussian (LoG) methods  
* **Batch processing**: Memory-optimized batch processing for large datasets
* **Validation**: Comprehensive parameter validation using Pydantic models

## Installation

```bash
pip install cryoblob
```

## Quick Start

```python
import cryoblob as cb

# Load an MRC file
mrc_image = cb.load_mrc("your_file.mrc")

# Process a folder of images
results = cb.folder_blobs("path/to/folder/")

# Plot results
cb.plot_mrc(mrc_image)
```

## Package Structure

The cryoblob package is organized into the following modules:

* **adapt**: Adaptive image processing with gradient descent optimization
* **blobs**: Core blob detection algorithms and preprocessing  
* **files**: File I/O operations and batch processing
* **image**: Basic image processing functions (filtering, resizing, etc.)
* **plots**: Visualization functions for MRC images and results
* **types**: Type definitions and PyTree structures
* **valid**: Parameter validation using Pydantic models

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