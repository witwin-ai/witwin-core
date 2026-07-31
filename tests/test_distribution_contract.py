from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
}


def test_core_source_tree_contains_no_native_code():
    native_sources = [
        path.relative_to(ROOT)
        for path in (ROOT / "witwin").rglob("*")
        if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES
    ]

    assert native_sources == []


def test_core_wheel_is_declared_as_pure_python():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build.hooks.custom]" not in pyproject
    assert "artifacts =" not in pyproject
