import ast
import enum
import inspect
from pathlib import Path

import mj_kdl_wrapper as mjk
from mj_kdl_wrapper import _mj_kdl_wrapper as ext

STUB = Path(mjk.__file__).with_name("_mj_kdl_wrapper.pyi")


def _stub_names() -> dict[str, set[str]]:
    tree = ast.parse(STUB.read_text())
    names: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    members.update(t.id for t in item.targets if isinstance(t, ast.Name))
            names[node.name] = members
        elif isinstance(node, ast.FunctionDef):
            names[node.name] = set()
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = set()
    # Drops stub-private helpers like _JointValues; keeps __version__ and friends.
    return {k: v for k, v in names.items() if not k.startswith("_") or k.startswith("__")}


def _public(names) -> set[str]:
    return {n for n in names if not n.startswith("_")}


def test_stub_and_module_name_the_same_api():
    stub = _stub_names()
    module = _public(dir(ext)) | {"__version__", "__mujoco_version__"}
    assert set(stub) - module == set(), "in the stub, missing from the module"
    assert module - set(stub) == set(), "in the module, missing from the stub"

    for cls_name, members in stub.items():
        cls = getattr(ext, cls_name)
        if not inspect.isclass(cls):
            continue
        if issubclass(cls, enum.Enum) or hasattr(cls, "__members__"):
            actual = set(cls.__members__)
        else:
            actual = _public(vars(cls))
        public = _public(members)
        assert public - actual == set(), f"{cls_name}: in the stub, missing from the module"
        assert actual - public == set(), f"{cls_name}: in the module, missing from the stub"
