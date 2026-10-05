from importlib.metadata import version

import mjkdl


def test_import():
    # The C++ build version (MJKDL_VERSION) must match the installed
    # package metadata, i.e. cmake/Versions.cmake and pyproject.toml agree.
    assert mjkdl.__version__ == version("mjkdl")
    assert mjkdl.LogLevel.ERROR.name == "ERROR"
    assert hasattr(mjkdl, "Viewer")
