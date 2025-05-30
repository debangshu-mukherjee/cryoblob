import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../.."))

on_rtd = os.environ.get("READTHEDOCS") == "True"

def read_pyproject_toml():
    pyproject_path = os.path.abspath("../../pyproject.toml")
    possible_paths = [
        pyproject_path,
        os.path.abspath(
            "../../../pyproject.toml"
        ),
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
        import tomllib
        with open(pyproject_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
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
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
            if name_match:
                data["project"]["name"] = name_match.group(1)
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                data["project"]["version"] = version_match.group(1)
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
                name_matches = re.findall(r'\{name\s*=\s*["\']([^"\']+)["\']', content)
                if name_matches:
                    data["project"]["authors"] = [
                        {"name": name} for name in name_matches
                    ]

        return data
    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return {"project": {}}


pyproject_data = read_pyproject_toml()
project_info = pyproject_data.get("project", {})
project = project_info.get("name", "cryoblob")
version = project_info.get("version", "unknown")
release = version
authors_list = project_info.get("authors", [])
if authors_list:
    author_names = [
        author.get("name", "") for author in authors_list if author.get("name")
    ]
    author = ", ".join(author_names)
else:
    author = "Debangshu Mukherjee, Alexis N. Williams"

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

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

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

if on_rtd:
    autodoc_mock_imports.extend(
        [
            "tomli",
            "tomllib",
        ]
    )

source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

master_doc = "index"

suppress_warnings = [
    'autodoc.import_error',
    'toc.not_readable',
    'docutils',
]