import os
import sys

# Add project root directory to sys.path (two levels up from docs/source)
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../.."))

project = "cryoblob"
copyright = "2024, Debangshu Mukherjee, Alexis N. Williams"
author = "Debangshu Mukherjee, Alexis N. Williams"
release = "2025.5.22"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = []

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Type hints
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# nbsphinx configuration for notebooks
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Intersphinx mapping for external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    'jax',
    'jax.numpy',
    'jaxtyping',
    'beartype',
    'mrcfile',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib_scalebar',
    'pandas',
    'tqdm',
    'pydantic',
    'chex',
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document (index page)
master_doc = 'index'