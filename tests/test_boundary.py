"""core/ must never import from domains/. This is the reusability guarantee."""
import ast
import pathlib

CORE = pathlib.Path(__file__).resolve().parents[1] / "core"


def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_does_not_import_domains():
    offenders = []
    for path in CORE.rglob("*.py"):
        for module in _imported_modules(path):
            if module.split(".")[0] == "domains":
                offenders.append(f"{path.name} imports {module}")
    assert not offenders, (
        "core/ must stay domain-agnostic, but found: " + "; ".join(offenders)
    )


def test_core_has_files_to_check():
    assert list(CORE.rglob("*.py")), "boundary test would vacuously pass"
