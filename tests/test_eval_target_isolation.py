"""
Evaluation-target isolation: the forward-looking realized variance in
volteq.rv.forward_target must be used only by evaluation code. Nothing under
models/, forecast/, or backtest/ may import or reference it. This test guards the
boundary now (those packages are empty) and as they fill in.
"""
import ast
import os

from volteq.config import repo_root
from volteq.rv import forward_target

GUARDED = ["models", "forecast", "backtest"]
NEEDLE = "forward_target"


def _py_files(pkg):
    root = os.path.join(repo_root(), "src", "volteq", pkg)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def test_guarded_packages_do_not_import_forward_target():
    offenders = []
    for pkg in GUARDED:
        for path in _py_files(pkg):
            with open(path) as fh:
                src = fh.read()
            # AST check: import statements
            for node in ast.walk(ast.parse(src, filename=path)):
                if isinstance(node, ast.ImportFrom) and node.module and NEEDLE in node.module:
                    offenders.append((path, f"from {node.module}"))
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if NEEDLE in a.name:
                            offenders.append((path, f"import {a.name}"))
            # raw-text check: dynamic import / attribute reference
            if NEEDLE in src:
                offenders.append((path, "textual reference"))
    assert not offenders, f"forward_target referenced by guarded code: {offenders}"


def test_forward_target_is_flagged_evaluation_only():
    assert getattr(forward_target, "EVALUATION_ONLY", False) is True
