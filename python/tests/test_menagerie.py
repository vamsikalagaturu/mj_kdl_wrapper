import importlib.util
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
    assert menagerie.asset_path("table.xml") == str(assets_dir / "table.xml")
    with pytest.raises(ValueError):
        menagerie.asset_path("../table.xml")
