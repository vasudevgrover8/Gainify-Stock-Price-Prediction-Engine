"""
Extract model architecture from legacy/yearly.py into modular model files.

This does NOT rewrite your core model.
It copies exact class/function bodies from legacy/yearly.py using Python AST.
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_YEARLY = PROJECT_ROOT / "legacy" / "yearly.py"
MODEL_DIR = PROJECT_ROOT / "src" / "gainify_stock_predictor" / "models"


source = LEGACY_YEARLY.read_text(encoding="utf-8")
tree = ast.parse(source)


def extract_node(name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise RuntimeError(f"Could not extract source for: {name}")
            return text.strip() + "\n"
    raise RuntimeError(f"Name not found in legacy/yearly.py: {name}")


def extract_existing(names):
    blocks = []
    missing = []

    for name in names:
        try:
            blocks.append(extract_node(name))
        except RuntimeError:
            missing.append(name)

    if missing:
        print("[WARN] Missing:", missing)

    return "\n\n".join(blocks)


def write_file(path: Path, header: str, names, footer: str = ""):
    body = extract_existing(names)

    content = header.rstrip() + "\n\n\n" + body.rstrip() + "\n"

    if footer.strip():
        content += "\n\n" + footer.strip() + "\n"

    path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {path.relative_to(PROJECT_ROOT)}")


# ------------------------------------------------------------
# 1. layers.py
# ------------------------------------------------------------
write_file(
    MODEL_DIR / "layers.py",
    '''"""
Custom TensorFlow/Keras layers.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import tensorflow as tf
from tensorflow.keras import layers
''',
    [
        "GatingLayer",
        "PositionalEncoding",
    ],
)


# ------------------------------------------------------------
# 2. cnn_encoder.py
# ------------------------------------------------------------
write_file(
    MODEL_DIR / "cnn_encoder.py",
    '''"""
Advanced CNN + attention encoder.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from configs.model_config import (
    L2REG,
    DROPOUT_ENC_PRETRAIN,
    DROPOUT_HEAD,
)

from gainify_stock_predictor.models.layers import PositionalEncoding
''',
    [
        "build_advanced_encoder",
    ],
)


# ------------------------------------------------------------
# 3. tft_blocks.py
# ------------------------------------------------------------
tft_blocks = '''"""
TFT-style block namespace.

Your original yearly.py keeps the TFT-like logic inside build_advanced_encoder():
- variable selection
- positional encoding
- multi-head attention stack
- gated feed-forward updates

No separate function existed originally, so this file is intentionally small.
"""


def tft_blocks_are_inside_encoder():
    """
    Return True to document that TFT-style blocks are implemented inside
    build_advanced_encoder(), preserving your original logic.
    """
    return True
'''

(MODEL_DIR / "tft_blocks.py").write_text(tft_blocks, encoding="utf-8")
print("[OK] Wrote src/gainify_stock_predictor/models/tft_blocks.py")


# ------------------------------------------------------------
# 4. attention_blocks.py
# ------------------------------------------------------------
attention_blocks = '''"""
Attention block namespace.

Your original yearly.py implements attention directly inside build_advanced_encoder()
using tensorflow.keras.layers.MultiHeadAttention.

No separate attention function existed originally.
"""


def attention_blocks_are_inside_encoder():
    """
    Return True to document that attention blocks are implemented inside
    build_advanced_encoder(), preserving your original logic.
    """
    return True
'''

(MODEL_DIR / "attention_blocks.py").write_text(attention_blocks, encoding="utf-8")
print("[OK] Wrote src/gainify_stock_predictor/models/attention_blocks.py")


# ------------------------------------------------------------
# 5. heads.py
# ------------------------------------------------------------
write_file(
    MODEL_DIR / "heads.py",
    '''"""
Model head functions.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from configs.model_config import (
    L2REG,
    DROPOUT_ENC_PRETRAIN,
    DROPOUT_ENC_FINETUNE,
    DROPOUT_HEAD,
)

from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.cnn_encoder import build_advanced_encoder
''',
    [
        "add_adapter",
        "_build_head",
    ],
)


# ------------------------------------------------------------
# 6. main_model.py
# ------------------------------------------------------------
write_file(
    MODEL_DIR / "main_model.py",
    '''"""
Main model builders.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

from configs.model_config import (
    DROPOUT_ENC_PRETRAIN,
    DROPOUT_ENC_FINETUNE,
)

from gainify_stock_predictor.models.heads import _build_head
''',
    [
        "build_pretrain_model",
        "build_multitask_model",
    ],
)


# ------------------------------------------------------------
# 7. tree_models.py
# ------------------------------------------------------------
write_file(
    MODEL_DIR / "tree_models.py",
    '''"""
Tree model helpers.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import lightgbm as lgb
import xgboost as xgb
''',
    [
        "build_tabular_dataset_1d",
        "train_lgbm_1d",
        "train_xgb_1d",
    ],
)


# ------------------------------------------------------------
# 8. models/__init__.py
# ------------------------------------------------------------
init_content = '''from gainify_stock_predictor.models.layers import (
    GatingLayer,
    PositionalEncoding,
)

from gainify_stock_predictor.models.cnn_encoder import build_advanced_encoder

from gainify_stock_predictor.models.tft_blocks import tft_blocks_are_inside_encoder
from gainify_stock_predictor.models.attention_blocks import attention_blocks_are_inside_encoder

from gainify_stock_predictor.models.heads import (
    add_adapter,
    _build_head,
)

from gainify_stock_predictor.models.main_model import (
    build_pretrain_model,
    build_multitask_model,
)

from gainify_stock_predictor.models.tree_models import (
    build_tabular_dataset_1d,
    train_lgbm_1d,
    train_xgb_1d,
)


__all__ = [
    "GatingLayer",
    "PositionalEncoding",
    "build_advanced_encoder",
    "tft_blocks_are_inside_encoder",
    "attention_blocks_are_inside_encoder",
    "add_adapter",
    "_build_head",
    "build_pretrain_model",
    "build_multitask_model",
    "build_tabular_dataset_1d",
    "train_lgbm_1d",
    "train_xgb_1d",
]
'''

(MODEL_DIR / "__init__.py").write_text(init_content, encoding="utf-8")
print("[OK] Updated src/gainify_stock_predictor/models/__init__.py")
