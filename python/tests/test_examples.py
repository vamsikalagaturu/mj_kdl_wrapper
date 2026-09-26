import subprocess
import sys
from pathlib import Path

import pytest

import mj_kdl_wrapper as mjk

EXAMPLES = Path(__file__).resolve().parents[1] / "mj_kdl_wrapper" / "examples"


@pytest.mark.parametrize("example", sorted(p.name for p in EXAMPLES.glob("ex_*.py")))
def test_example_runs_headless_and_meets_its_goal(example):
    try:
        mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / example)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
