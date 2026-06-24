from importlib.metadata import version

import mj_kdl_wrapper as mjk


def test_import():
    # The C++ build version (MJ_KDL_WRAPPER_VERSION) must match the installed
    # package metadata, i.e. cmake/Versions.cmake and pyproject.toml agree.
    assert mjk.__version__ == version("mj-kdl-wrapper")
    assert mjk.LogLevel.ERROR.name == "ERROR"
    assert hasattr(mjk, "SimulateViewer")
