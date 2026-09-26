import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

_MENAGERIE_PATH = Path(__file__).resolve().parents[1] / "mj_kdl_wrapper" / "menagerie.py"
_SPEC = importlib.util.spec_from_file_location("menagerie", _MENAGERIE_PATH)
assert _SPEC and _SPEC.loader
menagerie = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(menagerie)


def test_fetch_assets_to_cache_and_resolve(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)

    assets_dir = Path(menagerie.fetch_assets())

    assert assets_dir == tmp_path / "cache" / "mj_kdl_wrapper" / "assets"
    assert (assets_dir / "table.xml").exists()
    assert (assets_dir / "ft_sensor.xml").exists()
    assert menagerie.asset_path("table.xml") == str(assets_dir / "table.xml")
    with pytest.raises(ValueError):
        menagerie.asset_path("../table.xml")


def test_asset_path_copies_only_a_missing_or_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    table = Path(menagerie.asset_path("table.xml"))
    table.write_text("edited")
    assert Path(menagerie.asset_path("table.xml")).read_text() == "edited"
    (table.parent / menagerie._ASSETS_STAMP).write_text("stale")
    assert Path(menagerie.asset_path("table.xml")).read_text() != "edited"


def test_ur_models_resolve():
    for name in ("universal_robots_ur5e", "universal_robots_ur10e"):
        try:
            path = Path(menagerie.model_path(name))
        except RuntimeError as exc:
            pytest.skip(str(exc))
        assert path.name == name.removeprefix("universal_robots_") + ".xml"


@pytest.fixture
def git():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")


def test_fetch_refuses_a_directory_it_did_not_make(git, monkeypatch, tmp_path, capsys):
    dest = tmp_path / "mine"
    dest.mkdir()
    (dest / "notes.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="not a Menagerie checkout"):
        menagerie.fetch(dest)
    monkeypatch.setattr(sys, "argv", ["mj-kdl-fetch-menagerie", "--dest", str(dest)])
    assert menagerie.main() == 1
    assert "not a Menagerie checkout" in capsys.readouterr().err
    assert (dest / "notes.txt").read_text() == "keep"


def test_failed_fetch_leaves_nothing_and_says_why(git, monkeypatch, tmp_path):
    monkeypatch.setattr(menagerie, "MENAGERIE_REPO", str(tmp_path / "no-such-repo"))
    dest = tmp_path / "menagerie"
    with pytest.raises(RuntimeError, match=r"git fetch failed: \S"):
        menagerie.fetch(dest)
    assert list(tmp_path.iterdir()) == []
