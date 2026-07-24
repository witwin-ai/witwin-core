from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_world_contract_import_does_not_load_solver_or_native_runtime():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys
sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if type(finder).__module__ != "_witwin_channel_editable"
]
from witwin.core import DynamicScene, PhysicalMaterial, Scene, SceneSnapshot

blocked = (
    "rayd",
    "witwin.channel",
    "witwin.radar",
    "witwin.core.geometry.cuda",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)
)
if loaded:
    raise SystemExit("world import loaded forbidden modules: " + ", ".join(loaded))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
    )
