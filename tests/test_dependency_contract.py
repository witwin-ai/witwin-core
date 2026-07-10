from __future__ import annotations

import tomllib
from pathlib import Path


def test_runtime_and_optional_dependencies_do_not_include_slangtorch():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    groups = [pyproject["project"]["dependencies"]]
    groups.extend(pyproject["project"].get("optional-dependencies", {}).values())
    assert all(
        not dependency.lower().startswith("slangtorch")
        for group in groups
        for dependency in group
    )
