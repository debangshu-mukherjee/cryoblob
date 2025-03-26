"""
==================================================
JAX based differntiable data processing of cryo-EM 
particle images. Runs on both CPUs and GPUs.
==================================================

.. currentmodule:: arm_em

This package contains the functions needed to process the data, using cupy, 
for the cryo-EM particle images.
"""

from .blob_detection import *
from .image_utils import *
from .file_ops import *
