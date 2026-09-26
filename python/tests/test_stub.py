import ast
from pathlib import Path

import mj_kdl_wrapper as mjk
from mj_kdl_wrapper import _mj_kdl_wrapper as ext

STUB = Path(mjk.__file__).with_name("_mj_kdl_wrapper.pyi")
TREE = ast.parse(STUB.read_text())
STUB_CLASSES = {n.name: n for n in TREE.body if isinstance(n, ast.ClassDef)}


def _public(names) -> set[str]:
    return {n for n in names if not n.startswith("_")}


def _decorators(fn: ast.FunctionDef) -> set[str]:
    out = set()
    for d in fn.decorator_list:
        out.add(d.id if isinstance(d, ast.Name) else d.attr if isinstance(d, ast.Attribute) else "")
    return out


def _is_classvar(node: ast.AnnAssign) -> bool:
    ann = node.annotation
    return isinstance(ann, ast.Subscript) and getattr(ann.value, "id", "") == "ClassVar"


def _class_body(name: str) -> list[ast.stmt]:
    node = STUB_CLASSES[name]
    body = list(node.body)
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in STUB_CLASSES:
            body += _class_body(base.id)
    return body


def _members(name: str) -> dict[str, str]:
    """Member name -> 'method', 'static', 'readonly', 'writable' or 'classvar'."""
    out: dict[str, str] = {}
    for item in _class_body(name):
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            out[item.target.id] = "classvar" if _is_classvar(item) else "writable"
        elif isinstance(item, ast.FunctionDef):
            decorators = _decorators(item)
            if "setter" in decorators:
                out[item.name] = "writable"
            elif "property" in decorators:
                out.setdefault(item.name, "readonly")
            else:
                out[item.name] = "static" if "staticmethod" in decorators else "method"
    return out


def _split_top_level(text: str) -> list[str]:
    parts, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _runtime_params(obj) -> list[tuple[str, bool]]:
    """(name, has default) per parameter, from pybind11's signature line."""
    line = (obj.__doc__ or "").lstrip().splitlines()[0]
    assert not line.startswith("Overloaded"), f"{line}: overloads are not compared"
    start, depth = line.index("("), 0
    for end in range(start, len(line)):
        depth += {"(": 1, ")": -1}.get(line[end], 0)
        if depth == 0:
            break
    params = []
    for part in _split_top_level(line[start + 1 : end]):
        name = part.split(":")[0].split("=")[0].strip()
        if name != "/":
            params.append((name, " = " in part))
    return params


def _stub_params(fn: ast.FunctionDef) -> list[tuple[str, bool]]:
    args = fn.args.posonlyargs + fn.args.args
    n_plain = len(args) - len(fn.args.defaults)
    params = [(a.arg, i >= n_plain) for i, a in enumerate(args)]
    if fn.args.vararg:
        params.append(("*" + fn.args.vararg.arg, False))
    return params


def test_stub_and_module_name_the_same_api():
    stub = _public(STUB_CLASSES) | {
        n.name if isinstance(n, ast.FunctionDef) else n.target.id
        for n in TREE.body
        if isinstance(n, ast.FunctionDef)
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name))
    }
    module = _public(dir(ext)) | {"__version__", "__mujoco_version__"}
    assert stub - module == set(), "in the stub, missing from the module"
    assert module - stub == set(), "in the module, missing from the stub"


def test_stub_members_match_the_module():
    for name in _public(STUB_CLASSES):
        cls = getattr(ext, name)
        stub = _members(name)
        actual = _public(vars(cls))
        assert _public(stub) - actual == set(), f"{name}: in the stub, missing from the module"
        assert actual - _public(stub) == set(), f"{name}: in the module, missing from the stub"
        if "__members__" in stub:
            classvars = {m for m, kind in stub.items() if kind == "classvar"} - {"__members__"}
            assert classvars == set(cls.__members__), f"{name}: enum values differ"


def test_stub_read_only_properties_match_the_module():
    for name in _public(STUB_CLASSES):
        cls = getattr(ext, name)
        for member, kind in _members(name).items():
            if kind not in ("readonly", "writable"):
                continue
            attr = vars(cls).get(member)
            assert isinstance(attr, property), f"{name}.{member} is not a property"
            writable = attr.fset is not None
            assert writable == (kind == "writable"), (
                f"{name}.{member}: module {'writable' if writable else 'read-only'}, stub {kind}"
            )


def test_stub_signatures_match_the_module():
    functions = [(None, n) for n in TREE.body if isinstance(n, ast.FunctionDef)]
    for name in _public(STUB_CLASSES):
        functions += [
            (name, item)
            for item in _class_body(name)
            if isinstance(item, ast.FunctionDef) and not {"property", "setter"} & _decorators(item)
        ]
    for cls_name, fn in functions:
        owner = ext if cls_name is None else getattr(ext, cls_name)
        where = fn.name if cls_name is None else f"{cls_name}.{fn.name}"
        assert _runtime_params(getattr(owner, fn.name)) == _stub_params(fn), where
