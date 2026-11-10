from importlib.metadata import version

import mj_kdl_wrapper as mjk


def test_import():
    # The C++ build version (mj_kdl::version()) must match the installed
    # package metadata, i.e. cmake/Versions.cmake and pyproject.toml agree.
    assert mjk.__version__ == version("mj-kdl-wrapper")
    assert mjk.version().split("-")[0] == mjk.__version__
    assert mjk.LogLevel.ERROR.name == "ERROR"
    assert hasattr(mjk.Env, "overlay_geoms")
    # Python views and records through the C++ viewer and recorder.
    assert hasattr(mjk, "Viewer")
    assert hasattr(mjk, "VideoRecorder")
