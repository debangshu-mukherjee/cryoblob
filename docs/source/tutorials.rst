Tutorials
=========

Worked example notebooks demonstrating how to use cryoblob for cryo-EM image
processing and blob detection. Each notebook is self-contained; run them from the
``tutorials/`` directory of the repository.

.. rubric:: Notebook guide

- **1 — Laplacian of Gaussians**: the core scale-space LoG detector
  (``blob_list_log``) on a single micrograph.
- **2 — Folder blobs with LoG**: batch-process a folder of images with
  memory-aware batch sizing.
- **3 — Difference of Gaussians**: the DoG detector and how it compares to LoG.
- **4 — Image stacks**: loading and processing multi-frame / stacked data.
- **5 — Particle sizes (PLGA)**: measuring particle-size distributions from
  detected blobs.

.. toctree::
   :maxdepth: 1
   :caption: Example notebooks

   tutorials/1_laplacian_of_gaussians
   tutorials/2_folder_blobs_with_log
   tutorials/3_difference_of_gaussians
   tutorials/4_image_stacks
   tutorials/5_Particle_Sizes_PLGA

.. toctree::
   :maxdepth: 1
   :caption: Feature guides

   overlap_and_plotting
