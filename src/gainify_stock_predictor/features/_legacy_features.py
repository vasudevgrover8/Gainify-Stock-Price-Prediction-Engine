"""
Lazy loader for original legacy/yearly.py.

This file does not rewrite feature logic.
It loads the original yearly.py only when a legacy feature function is called.
"""

import importlib.util
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_YEARLY_PATH = PROJECT_ROOT / "legacy" / "yearly.py"


@lru_cache(maxsize=1)
def load_legacy_yearly():
    """
    Load legacy/yearly.py as a Python module.

    This keeps yearly.py as the source of truth.
    """
    if not LEGACY_YEARLY_PATH.exists():
        raise FileNotFoundError(f"legacy/yearly.py not found at: {LEGACY_YEARLY_PATH}")

    spec = importlib.util.spec_from_file_location("legacy_yearly", LEGACY_YEARLY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_legacy_function(function_name, *args, **kwargs):
    """
    Call a function from legacy/yearly.py by name.
    """
    module = load_legacy_yearly()

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in legacy/yearly.py")

    return getattr(module, function_name)(*args, **kwargs)


def get_legacy_function(function_name):
    """
    Return a function object from legacy/yearly.py.
    """
    module = load_legacy_yearly()

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in legacy/yearly.py")

    return getattr(module, function_name)
