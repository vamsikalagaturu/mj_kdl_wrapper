from importlib.metadata import version

import mj_kdl_wrapper as mjk


def test_import():
    # cmake/Versions.cmake and pyproject.toml must agree.
    assert mjk.__version__ == version("mj-kdl-wrapper")
    assert mjk.version().split("-")[0] == mjk.__version__
    assert mjk.LogLevel.ERROR.name == "ERROR"
    assert hasattr(mjk.Env, "overlay_geoms")
    assert hasattr(mjk, "Viewer")
    assert hasattr(mjk, "VideoRecorder")
