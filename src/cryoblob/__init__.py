"""
Module: cryoblob
---------------------------

JAX based, JIT compiled, scalable codes for
detection of amorphous blobs in low SNR cryo-EM
images.

Submodules
----------
- `adapt`:
    Adaptive image processing methods that take
    advantage of JAX's automatic differentiation capabilities.
- `blobs`:
    Contains the core blob detection algorithms.
- `files`:
    Interfacing with data files.
- `image`:
    Utility functions for image processing.
- `plots`:
    Plotting functions for visualizing MRC images
    and blob detection results.
- `types`:
    Type aliases and PyTrees.
- `valid`:
    Pydantic models for data validation and configuration management.
"""

from .adapt import *
from .blobs import *
from .files import *
from .image import *
from .plots import *
from .types import *
from .valid import *
