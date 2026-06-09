"""
Rewrite training/losses.py correctly from legacy/yearly.py.

Fix:
- loss functions are written first
- loss dictionaries/weights are written after the functions
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_YEARLY = PROJECT_ROOT / "legacy" / "yearly.py"
LOSSES_FILE = PROJECT_ROOT / "src" / "gainify_stock_predictor" / "training" / "losses.py"

source = LEGACY_YEARLY.read_text(encoding="utf-8")
tree = ast.parse(source)


FUNCTIONS = [
    "spike_weighted_mse",
    "smooth_huber",
    "focal_bce_soft",
]

ASSIGNMENTS = [
    "LOSSES",
    "LOSS_W_PRE",
    "LOSS_W_FT_M",
    "LOSS_W_FT",
    "LOSS_W_DAILY",
]


def extract_function(name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise RuntimeError(f"Could not extract function: {name}")
            return text.strip()
    raise RuntimeError(f"Function not found: {name}")


def extract_assignment(name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    text = ast.get_source_segment(source, node)
                    if text is None:
                        raise RuntimeError(f"Could not extract assignment: {name}")
                    return text.strip()
    raise RuntimeError(f"Assignment not found: {name}")


parts = [
'''"""
Loss functions and loss weights.

Physically extracted from legacy/yearly.py.
Original loss logic and weights are preserved.

Important:
Functions must be defined before LOSSES dictionary.
"""

import tensorflow as tf
'''
]

for fn in FUNCTIONS:
    parts.append(extract_function(fn))

for assignment in ASSIGNMENTS:
    parts.append(extract_assignment(assignment))

LOSSES_FILE.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")

print("[OK] Rewrote training/losses.py with correct order")
