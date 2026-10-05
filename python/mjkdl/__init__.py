from importlib.resources import files as _files
from pathlib import Path as _Path

import mujoco as _mujoco  # noqa: F401  # registers its plugins first; after the wrapper's it aborts

from . import _mjkdl
from ._mjkdl import *

__version__ = _mjkdl.__version__
__mujoco_version__ = _mjkdl.__mujoco_version__

# The bundled models (Gen3, 2F-85, table, cabinet, F/T sensor, ...) installed with the package.
ASSETS_DIR = _Path(str(_files(__name__) / "assets"))
