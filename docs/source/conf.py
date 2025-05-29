import os
import sys
import re
from datetime import datetime

# Add project root directory to sys.path (two levels up from docs/source)
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../.."))

# Read project metadata from pyproject.toml
def read_pyproject_toml():
    pyproject_path = os.path.abspath("../../pyproject.toml")
    try:
        # Try tomllib first (Python 3.11+)
        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            # Fall back to tomli
            try:
                import tomli
                with open(pyproject_path, "rb") as f:
                    return tomli.load(f)
            except ImportError:
                # Manual parsing as last resort
                return parse_pyproject_manually(pyproject_path)
    except FileNotFoundError:
        print(f"Warning: Could not find pyproject.toml at {pyproject_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not parse pyproject.toml: {e}")
        return {}

def parse_pyproject_manually(filepath):
    """Simple manual parser for basic pyproject.toml values"""
    data = {"project": {}}
    current_section = None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Section headers
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    continue
                
                # Only parse [project] section
                if current_section == "project":
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        if key == "name":
                            data["project"]["name"] = value
                        elif key == "version":
                            data["project"]["version"] = value
                        elif key == "authors":
                            # Simple parsing of authors array
                            authors_match = re.findall(r'name\s*=\s*["\']([^"\']+)["\']', line)
                            if authors_match:
                                data["project"]["authors"] = [{"name": name} for name in authors_match]
                            else:
                                # Try to extract from the whole authors line
                                authors_line = value
                                names = re.findall(r'name\s*=\s*["\']([^"\']+)["\']', authors_line)
                                if names:
                                    data["project"]["authors"] = [{"name": name} for name in names]
        
        return data
    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return {"project": {}}

pyproject_data = read_pyproject_toml()
project_info = pyproject_data.get("project", {})

# Extract project information
project = project_info.get("name", "cryoblob")
version = project_info.get("version", "unknown")
release = version

# Extract authors
authors_list = project_info.get("authors", [])
if authors_list:
    author_names = [author.get("name", "") for author in authors_list if author.get("name")]
    author = ", ".join(author_names)
else:
    author = "Unknown"

# Set copyright with current year
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

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