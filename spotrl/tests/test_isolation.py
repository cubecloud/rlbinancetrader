"""Гейт Э0.3 — изоляция пакета от старого кода.

Проверка: ни один модуль `spotrl` не импортирует `binanceenv` и
`rllab.rllaboratory`. Старые модули при этом продолжают жить в репозитории
и импортироваться сами по себе — этот тест их не трогает.
"""
from __future__ import annotations

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("binanceenv", "rllab.rllaboratory")


def test_no_forbidden_imports():
    """grep по пакету: ни одного импорта запрещённых модулей."""
    pattern = re.compile(r"^\s*(?:from|import)\s+(" + "|".join(
        re.escape(name) for name in FORBIDDEN) + r")\b", re.MULTILINE)
    hits = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            hits.append(str(path))
    assert not hits, f"запрещённые импорты в: {hits}"


def test_public_objects_have_docstrings():
    """Docstring есть у каждого публичного объекта пакета (PLAN 1.5.4)."""
    import ast

    missing = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(f"{path}::<module>")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue
                if ast.get_docstring(node) is None:
                    missing.append(f"{path}::{node.name}")
    assert not missing, f"без docstring: {missing}"
