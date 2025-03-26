from beartype.typing import Tuple, List
import os
import json
from importlib.resources import files

def file_params() -> Tuple[str, dict]:
    """
    Description
    -----------
    Run this at the beginning to generate the dict
    This gives both the absolute and relative paths
    on how the files are organized.

    Returns
    -------
    - `main_directory` (str):
        the main directory where the package is located.
    - `folder_structure` (dict):
        where the files and data are stored, as read
        from the organization.json file.
    """
    pkg_directory: str = os.path.dirname(__file__)
    listring: List = pkg_directory.split("/")[1:-2]
    listring.append("")
    listring.insert(0, "")
    main_directory: str = "/".join(listring)
    folder_structure: dict = json.load(
        open(files("arm_em.params").joinpath("organization.json"))
    )
    return (main_directory, folder_structure)