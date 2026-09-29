from mj_kdl_wrapper import fetch_examples


def test_copies_the_examples_without_run_leftovers(tmp_path):
    fetch_examples.copy_examples(tmp_path)

    examples = tmp_path / "examples"
    assert (examples / "ex_gravity_comp.py").exists()
    leftovers = [p for p in examples.rglob("*") if p.name in ("__pycache__", "MUJOCO_LOG.TXT")]
    assert leftovers == []
