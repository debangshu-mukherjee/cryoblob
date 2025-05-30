import os
import re
import sys
from datetime import datetime

# Add project root directory to sys.path (two levels up from docs/source)
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../.."))

# Check if we're on Read the Docs
on_rtd = os.environ.get("READTHEDOCS") == "True"


# Read project metadata from pyproject.toml
def read_pyproject_toml():
    pyproject_path = os.path.abspath("../../pyproject.toml")

    # If pyproject.toml doesn't exist, try different locations
    possible_paths = [
        pyproject_path,
        os.path.abspath(
            "../../../pyproject.toml"
        ),  # In case we're in a different structure
        os.path.abspath("./pyproject.toml"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            pyproject_path = path
            break
    else:
        print(f"Warning: Could not find pyproject.toml in any expected location")
        return {}

    try:
        # Use tomllib (Python 3.11+)
        import tomllib

        with open(pyproject_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        # Fall back to manual parsing for older Python versions
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
    authors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

            # Extract name
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
            if name_match:
                data["project"]["name"] = name_match.group(1)

            # Extract version
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                data["project"]["version"] = version_match.group(1)

            # Extract authors
            authors_matches = re.findall(
                r'name\s*=\s*["\']([^"\']+)["\'].*?email\s*=\s*["\']([^"\']+)["\']',
                content,
                re.DOTALL,
            )
            if authors_matches:
                data["project"]["authors"] = [
                    {"name": name, "email": email} for name, email in authors_matches
                ]
            else:
                # Try just names
                name_matches = re.findall(r'\{name\s*=\s*["\']([^"\']+)["\']', content)
                if name_matches:
                    data["project"]["authors"] = [
                        {"name": name} for name in name_matches
                    ]

        return data
    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return {"project": {}}


# Get project metadata
pyproject_data = read_pyproject_toml()
project_info = pyproject_data.get("project", {})

# Extract project information with fallbacks
project = project_info.get("name", "cryoblob")
version = project_info.get("version", "unknown")
release = version

# Extract authors
authors_list = project_info.get("authors", [])
if authors_list:
    author_names = [
        author.get("name", "") for author in authors_list if author.get("name")
    ]
    author = ", ".join(author_names)
else:
    author = "Debangshu Mukherjee, Alexis N. Williams"  # Fallback

# Set copyright with current year
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

print(f"Building docs for {project} v{version} by {author}")

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
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type hints
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# nbsphinx configuration for notebooks
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Intersphinx mapping for external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    "jax",
    "jax.numpy",
    "jax.scipy",
    "jax.scipy.signal",
    "jax.tree_util",
    "jax.lax",
    "jaxtyping",
    "beartype",
    "beartype.typing",
    "mrcfile",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.axes",
    "matplotlib.figure",
    "matplotlib_scalebar",
    "matplotlib_scalebar.scalebar",
    "pandas",
    "tqdm",
    "tqdm.auto",
    "pydantic",
    "pydantic.types",
    "chex",
    "numpy",
    "absl",
    "absl.testing",
]

# Additional RTD-specific mocking
if on_rtd:
    autodoc_mock_imports.extend(
        [
            "tomli",
            "tomllib",
        ]
    )

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# Master document (index page)
master_doc = "index"
