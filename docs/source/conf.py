import os
import sys

# Add project root directory to sys.path (two levels up)
sys.path.insert(0, os.path.abspath("../.."))

project = 'cryoblob'
copyright = '2024'
author = 'Your Name'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']

# nbsphinx configuration for notebooks outside docs/source
nbsphinx_execute = 'auto'  # 'never' if no auto execution desired

# Explicit mapping of notebooks from top-level tutorials folder
nbsphinx_notebooks = {
    "tutorials/notebook1": "../../tutorials/notebook1.ipynb",
    "tutorials/notebook2": "../../tutorials/notebook2.ipynb",
    # add further notebooks here explicitly
}
