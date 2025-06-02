API Reference
=============

This page contains the API reference for all modules in the cryoblob package.

Core Detection Modules
----------------------

Traditional Blob Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cryoblob.blobs
   :members:
   :undoc-members:
   :show-inheritance:

Multi-method Enhanced Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cryoblob.multi
   :members:
   :undoc-members:
   :show-inheritance:

Image Processing and Utilities
------------------------------

Image Processing
~~~~~~~~~~~~~~~

.. automodule:: cryoblob.image
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Processing
~~~~~~~~~~~~~~~~~~

.. automodule:: cryoblob.adapt
   :members:
   :undoc-members:
   :show-inheritance:

Data I/O and Visualization
--------------------------

File Operations
~~~~~~~~~~~~~~

.. automodule:: cryoblob.files
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: cryoblob.plots
   :members:
   :undoc-members:
   :show-inheritance:

Configuration and Types
-----------------------

Parameter Validation
~~~~~~~~~~~~~~~~~~~

.. automodule:: cryoblob.valid
   :members:
   :undoc-members:
   :show-inheritance:

Type Definitions
~~~~~~~~~~~~~~~

.. automodule:: cryoblob.types
   :members:
   :undoc-members:
   :show-inheritance:

Detection Method Guide
---------------------

Method Selection
~~~~~~~~~~~~~~~

+------------------+-------------------+------------------------+------------------+
| Blob Type        | Recommended       | Alternative Methods    | Key Parameters   |
|                  | Method            |                        |                  |
+==================+===================+========================+==================+
| Circular         | ``blob_list_log`` | ``hessian_blob_``      | ``min_blob_size``|
|                  |                   | ``detection``          | ``max_blob_size``|
+------------------+-------------------+------------------------+------------------+
| Elongated        | ``ridge_``        | ``multi_scale_ridge_`` | ``ridge_``       |
| (pill-shaped)    | ``detection``     | ``detector``           | ``threshold``    |
+------------------+-------------------+------------------------+------------------+
| Overlapping      | ``watershed_``    | ``enhanced_blob_``     | ``min_marker_``  |
|                  | ``segmentation``  | ``detection`` (full)   | ``distance``     |
+------------------+-------------------+------------------------+------------------+
| Mixed/Complex    | ``enhanced_blob_``| Factory functions      | Use predefined   |
|                  | ``detection``     | from ``valid`` module  | configurations   |
+------------------+-------------------+------------------------+------------------+

Key Functions by Category
~~~~~~~~~~~~~~~~~~~~~~~~

**Basic Detection**

* ``blob_list_log()`` - Traditional LoG-based circular blob detection
* ``preprocessing()`` - Image preprocessing pipeline

**Enhanced Detection**

* ``enhanced_blob_detection()`` - Multi-method comprehensive detection
* ``hessian_blob_detection()`` - Hessian-based blob detection with superior boundaries
* ``ridge_detection()`` - Detect elongated objects using eigenvalue analysis
* ``watershed_segmentation()`` - Separate overlapping structures

**Configuration Factories**

* ``create_default_pipeline()`` - Standard detection configuration
* ``create_elongated_objects_pipeline()`` - Optimized for elongated objects
* ``create_overlapping_blobs_pipeline()`` - Optimized for overlapping structures
* ``create_comprehensive_pipeline()`` - All methods enabled

Quick Usage Examples
~~~~~~~~~~~~~~~~~~~

**Circular Blobs**

.. code-block:: python

   blobs = cb.blob_list_log(mrc_image, min_blob_size=5, max_blob_size=20)

**Elongated Objects**

.. code-block:: python

   config = cb.create_elongated_objects_pipeline()
   _, elongated, _ = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())

**Overlapping Structures**

.. code-block:: python

   config = cb.create_overlapping_blobs_pipeline()
   circular, _, separated = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())

**Comprehensive Analysis**

.. code-block:: python

   config = cb.create_comprehensive_pipeline()
   circular, elongated, overlapping = cb.enhanced_blob_detection(mrc_image, **config.to_enhanced_kwargs())