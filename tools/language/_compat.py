"""Load models/motion/meta_action.py by FILE PATH so the language tools run on a laptop without mmcv / mmdet3d.

If mmcv is installed (repo env) the real BaseModule / PLUGIN_LAYERS are used; otherwise tiny stand-ins are
injected, and the real meta_action.py is still the code that runs.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_mmcv_stub():
    try:
        import mmcv.cnn.bricks.registry  # noqa: F401
        import mmcv.runner.base_module  # noqa: F401

        return
    except Exception:
        pass

    class _Registry:
        def register_module(self, *a, **k):
            return lambda cls: cls

    for name in ("mmcv", "mmcv.runner", "mmcv.runner.base_module", "mmcv.cnn", "mmcv.cnn.bricks", "mmcv.cnn.bricks.registry"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["mmcv.runner.base_module"].BaseModule = nn.Module
    sys.modules["mmcv.cnn.bricks.registry"].PLUGIN_LAYERS = _Registry()


def load_meta_action_module(repo_root=None):
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    path = repo_root / "projects" / "mmdet3d_plugin" / "models" / "motion" / "meta_action.py"
    _ensure_mmcv_stub()
    spec = importlib.util.spec_from_file_location("meta_action_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_language_package(repo_root=None, name="lang_pkg"):
    """Import models/language as a stand-alone package (does not import the heavy models/__init__)."""
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    pkg_dir = repo_root / "projects" / "mmdet3d_plugin" / "models" / "language"
    spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
