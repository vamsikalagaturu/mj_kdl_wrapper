import mj_kdl_wrapper as mjk


def test_import():
    assert mjk.__version__ == "0.1.0"
    assert mjk.LogLevel.ERROR.name == "ERROR"
    assert hasattr(mjk, "SimulateViewer")
