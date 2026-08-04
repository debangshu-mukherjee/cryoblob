Overlap recovery and plotting
=============================

This guide covers two features added in 2026.7.0: an overlap-recovery detector
(:func:`cryoblob.blob_list_log_watershed`) and blob overlays in
:func:`cryoblob.plot_mrc`.

Basic detection with an overlay
-------------------------------

``blob_list_log`` returns an ``(n, 3)`` array of ``(Y, X, size)`` in physical
units. Pass it straight to ``plot_mrc`` to draw the detections on the image:

.. code-block:: python

   import cryoblob as cb

   mrc_image = cb.load_mrc("micrograph_0001.mrc")
   blobs = cb.blob_list_log(mrc_image, min_blob_size=8, max_blob_size=110)

   # Overlay the detected blobs (circles at the detected positions and radii)
   cb.plot_mrc(mrc_image, blobs=blobs, blob_color="cyan")

The physical blob coordinates are converted back to pixel coordinates using the
image's voxel size, so the overlay aligns with the displayed micrograph.

Recovering overlapping particles
--------------------------------

Plain LoG merges touching particles into a single detection. For dense fields,
:func:`cryoblob.blob_list_log_watershed` runs LoG detection followed by a
distance-transform watershed pass that **keeps every LoG detection and adds only
watershed detections not already found** — recovering merged particles while
preserving LoG precision:

.. code-block:: python

   blobs = cb.blob_list_log_watershed(
       mrc_image, min_blob_size=8, max_blob_size=110
   )
   cb.plot_mrc(mrc_image, blobs=blobs)

On dense/overlapping fields this typically increases recall at essentially
unchanged precision; on well-separated fields the effect is negligible, so it is
safe to use as a drop-in replacement for ``blob_list_log`` when overlap is
expected. Unlike ``enhanced_blob_detection`` (Hessian-seeded watershed, which can
over-split), this mode is LoG-seeded.

When to use which
-----------------

+-------------------------------+------------------------------------------+
| Situation                     | Recommended                              |
+===============================+==========================================+
| Well-separated particles      | ``blob_list_log``                        |
+-------------------------------+------------------------------------------+
| Touching / overlapping        | ``blob_list_log_watershed``              |
| particles                     |                                          |
+-------------------------------+------------------------------------------+
| Visual QC of any result       | ``plot_mrc(..., blobs=blobs)``           |
+-------------------------------+------------------------------------------+

Both detectors share the same parameters (``min_blob_size``, ``max_blob_size``,
``blob_step``, ``downscale``, ``std_threshold``); see the API reference for
details.
