from importlib.resources import files as _files
from pathlib import Path as _Path

import mujoco as _mujoco  # noqa: F401  # registers its plugins first; after the wrapper's it aborts

from . import _mj_kdl_wrapper
from ._mj_kdl_wrapper import *

__version__ = _mj_kdl_wrapper.__version__
__mujoco_version__ = _mj_kdl_wrapper.__mujoco_version__

# The bundled models (Gen3, 2F-85, table, cabinet, F/T sensor, ...) installed with the package.
ASSETS_DIR = _Path(str(_files(__name__) / "assets"))
