cryoblob Documentation
======================

**cryoblob** is a JAX-based, JIT-compiled, scalable package for detection of amorphous blobs in low SNR cryo-EM images.

Features
--------

* **JAX-powered**: Leverages JAX for high-performance computing with automatic differentiation
* **GPU acceleration**: Can utilize both CPUs and GPUs for processing
* **Adaptive filtering**: Includes adaptive Wiener filtering and thresholding
* **Blob detection**: Advanced blob detection using Laplacian of Gaussian (LoG) methods  
* **Batch processing**: Memory-optimized batch processing for large datasets
* **Validation**: Comprehensive parameter validation using Pydantic models

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install -e .

Basic usage:

.. code-block:: python

   import cryoblob as cb
   
   # Load an MRC file
   mrc_image = cb.load_mrc("your_file.mrc")
   
   # Process a folder of images
   results = cb.folder_blobs("path/to/folder/")

Package Structure
-----------------

The cryoblob package is organized into the following modules:

* **adapt**: Adaptive image processing with gradient descent optimization
* **blobs**: Core blob detection algorithms and preprocessing  
* **files**: File I/O operations and batch processing
* **image**: Basic image processing functions (filtering, resizing, etc.)
* **plots**: Visualization functions for MRC images and results
* **types**: Type definitions and PyTree structures
* **valid**: Parameter validation using Pydantic models

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference
   tutorials

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`