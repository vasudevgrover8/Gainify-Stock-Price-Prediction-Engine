"""
Extract training, checkpoint, prediction, and reporting functions from legacy/yearly.py.

This copies exact function bodies from legacy/yearly.py using Python AST.
It does not rewrite your training stages, checkpoint hierarchy, prediction logic, or reports.
"""

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_YEARLY = PROJECT_ROOT / "legacy" / "yearly.py"
SRC_ROOT = PROJECT_ROOT / "src" / "gainify_stock_predictor"

TRAINING_DIR = SRC_ROOT / "training"
CHECKPOINT_DIR = SRC_ROOT / "checkpoints"
PREDICTION_DIR = SRC_ROOT / "prediction"
REPORTING_DIR = SRC_ROOT / "reporting"


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


def extract_assignment(name: str) -> str:
    pattern = rf"^{name}\s*=.*$"
    m = re.search(pattern, source, flags=re.MULTILINE)

    if not m:
        print(f"[WARN] Assignment not found: {name}")
        return ""

    return m.group(0)


def write_file(path: Path, header: str, names=None, assignments=None, footer: str = ""):
    names = names or []
    assignments = assignments or []

    assignment_text = "\n".join([extract_assignment(a) for a in assignments if extract_assignment(a)])

    body = extract_existing(names)

    content = header.rstrip() + "\n\n"

    if assignment_text.strip():
        content += assignment_text.strip() + "\n\n"

    if body.strip():
        content += body.strip() + "\n"

    if footer.strip():
        content += "\n\n" + footer.strip() + "\n"

    path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {path.relative_to(PROJECT_ROOT)}")


# ============================================================
# CHECKPOINTS
# ============================================================

write_file(
    CHECKPOINT_DIR / "checkpoint_manager.py",
    '''"""
Checkpoint path and loading helpers.

Physically extracted from legacy/yearly.py.
Original checkpoint hierarchy is preserved.
"""

import os
import json
import logging
from datetime import datetime

from configs.paths_config import MODEL_DIR, STAGE_DIRS, STAGE_PARENT


log = logging.getLogger(__name__)
''',
    [
        "get_stage_output_dir",
        "load_latest_successful_checkpoint",
        "resolve_parent_checkpoint",
    ],
)


write_file(
    CHECKPOINT_DIR / "metadata_manager.py",
    '''"""
Checkpoint metadata helpers.

Physically extracted from legacy/yearly.py.
Original metadata fields are preserved.
"""

import os
import json
from datetime import datetime
''',
    [
        "save_stage_metadata",
    ],
)


(CHECKPOINT_DIR / "__init__.py").write_text(
'''"""
Checkpoint package.

Heavy imports are intentionally avoided here.
Import directly from checkpoint_manager.py or metadata_manager.py when needed.
"""
''',
    encoding="utf-8",
)


# ============================================================
# TRAINING
# ============================================================

write_file(
    TRAINING_DIR / "losses.py",
    '''"""
Loss functions and loss weights.

Physically extracted from legacy/yearly.py.
Original loss logic and weights are preserved.
"""

import tensorflow as tf
''',
    [
        "spike_weighted_mse",
        "smooth_huber",
        "focal_bce_soft",
    ],
    [
        "LOSSES",
        "LOSS_W_PRE",
        "LOSS_W_FT_M",
        "LOSS_W_FT",
        "LOSS_W_DAILY",
    ],
)


write_file(
    TRAINING_DIR / "trainer_utils.py",
    '''"""
Training utility functions.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.model_config import SEQ_LEN, EMBARGO_STEPS, LABEL_SMOOTH_EPS
from configs.training_config import (
    MONTHLY_LOOKBACK_DAYS,
    WEEKLY_LOOKBACK_DAYS,
    DAILY_LOOKBACK_DAYS,
)

from gainify_stock_predictor.preprocessing.sequence_builder import (
    cumulative_logret_forward,
    make_sequences_masked,
    apply_label_smoothing,
)

from gainify_stock_predictor.features.feature_pipeline import build_features_from_df


log = logging.getLogger(__name__)
''',
    [
        "merge_small_buckets",
        "compute_beta",
        "compute_momentum_score",
        "compute_mean_reversion_score",
        "make_dir_labels_1d",
        "build_and_save_sequences_for_stock",
        "prepare_single_stock_arrays",
        "prepare_daily_arrays",
    ],
)


write_file(
    TRAINING_DIR / "yearly_pretrain.py",
    '''"""
Yearly pretraining stage.

Physically extracted from legacy/yearly.py.
Original yearly pretrain logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN
from configs.training_config import EPOCHS_PT, BATCH, LR_PT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_pretrain_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_PRE
from gainify_stock_predictor.training.trainer_utils import build_and_save_sequences_for_stock
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata


log = logging.getLogger(__name__)
''',
    [
        "pretrain_bucket",
        "run_yearly_pretrain",
    ],
)


write_file(
    TRAINING_DIR / "monthly_finetune.py",
    '''"""
Monthly fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original monthly fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_MONTHLY_FT, LR_MONTHLY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_FT_M
from gainify_stock_predictor.training.trainer_utils import prepare_single_stock_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)
''',
    [
        "run_monthly_finetune",
    ],
)


write_file(
    TRAINING_DIR / "weekly_finetune.py",
    '''"""
Weekly fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original weekly fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_WEEKLY_FT, LR_WEEKLY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_FT
from gainify_stock_predictor.training.trainer_utils import prepare_single_stock_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)
''',
    [
        "run_weekly_finetune",
    ],
)


write_file(
    TRAINING_DIR / "daily_finetune.py",
    '''"""
Daily fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original daily fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_DAILY_FT, LR_DAILY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.models.layers import GatingLayer, PositionalEncoding
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_DAILY
from gainify_stock_predictor.training.trainer_utils import prepare_daily_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)
''',
    [
        "run_daily_finetune",
    ],
)


(TRAINING_DIR / "callbacks.py").write_text(
'''"""
Callback helper namespace.

Your original yearly.py creates callbacks directly inside each training stage.
This file is kept for project structure compatibility.
"""


def callbacks_are_defined_inside_stage_functions():
    """
    Returns True because the original callback logic is preserved inside:
    - run_yearly_pretrain
    - run_monthly_finetune
    - run_weekly_finetune
    - run_daily_finetune
    """
    return True
''',
    encoding="utf-8",
)


(TRAINING_DIR / "__init__.py").write_text(
'''"""
Training package.

Heavy TensorFlow imports are intentionally avoided here.
Import directly from the stage files when running training.
"""
''',
    encoding="utf-8",
)


# ============================================================
# PREDICTION
# ============================================================

write_file(
    PREDICTION_DIR / "predictor.py",
    '''"""
Prediction helpers.

Physically extracted from legacy/yearly.py.
Original forecast logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.models.layers import GatingLayer, PositionalEncoding
from gainify_stock_predictor.checkpoints.checkpoint_manager import load_latest_successful_checkpoint


log = logging.getLogger(__name__)
''',
    [
        "forecast_1d",
        "run_predict_only",
    ],
)


write_file(
    PREDICTION_DIR / "signal_builder.py",
    '''"""
Signal and verdict builders.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd
''',
    [
        "calculate_verdict",
        "ensemble_next_day_signal",
    ],
)


(PREDICTION_DIR / "ranker.py").write_text(
'''"""
Prediction ranking namespace.

Ranking reports are implemented in reporting/ranked_reports.py.
This file exists to keep the requested project structure.
"""


def ranker_is_report_based():
    """
    Ranking is handled through reporting.build_ranked_volatility_reports().
    """
    return True
''',
    encoding="utf-8",
)


(PREDICTION_DIR / "__init__.py").write_text(
'''"""
Prediction package.

Heavy imports are intentionally avoided here.
Import directly from predictor.py or signal_builder.py when needed.
"""
''',
    encoding="utf-8",
)


# ============================================================
# REPORTING
# ============================================================

write_file(
    REPORTING_DIR / "prediction_writer.py",
    '''"""
Prediction CSV writer.

Physically extracted from legacy/yearly.py.
Original output format is preserved.
"""

import os
import logging

import pandas as pd

from configs.paths_config import OUTPUT_DIR


log = logging.getLogger(__name__)
''',
    [
        "save_master_predictions_csv",
    ],
)


write_file(
    REPORTING_DIR / "ranked_reports.py",
    '''"""
Ranked volatility report builder.

Physically extracted from legacy/yearly.py.
Original report logic is preserved.
"""

import os
import logging

import pandas as pd

from configs.paths_config import OUTPUT_DIR


log = logging.getLogger(__name__)
''',
    [
        "build_ranked_volatility_reports",
    ],
)


write_file(
    REPORTING_DIR / "metrics.py",
    '''"""
Evaluation metric helpers.

Physically extracted from legacy/yearly.py.
Original evaluation logic is preserved.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
''',
    [
        "evaluate_holdout_close",
    ],
)


(REPORTING_DIR / "__init__.py").write_text(
'''"""
Reporting package.

Import directly from prediction_writer.py, ranked_reports.py, or metrics.py.
"""
''',
    encoding="utf-8",
)


# ============================================================
# TESTS
# ============================================================

test_content = '''from pathlib import Path


def test_training_files_exist():
    root = Path(__file__).resolve().parents[1]
    training_dir = root / "src" / "gainify_stock_predictor" / "training"

    required = [
        "yearly_pretrain.py",
        "monthly_finetune.py",
        "weekly_finetune.py",
        "daily_finetune.py",
        "trainer_utils.py",
        "losses.py",
        "callbacks.py",
        "__init__.py",
    ]

    for file_name in required:
        assert (training_dir / file_name).exists(), f"Missing training file: {file_name}"


def test_checkpoint_files_exist():
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "src" / "gainify_stock_predictor" / "checkpoints"

    required = [
        "checkpoint_manager.py",
        "metadata_manager.py",
        "__init__.py",
    ]

    for file_name in required:
        assert (checkpoint_dir / file_name).exists(), f"Missing checkpoint file: {file_name}"


def test_prediction_reporting_files_exist():
    root = Path(__file__).resolve().parents[1]

    prediction_dir = root / "src" / "gainify_stock_predictor" / "prediction"
    reporting_dir = root / "src" / "gainify_stock_predictor" / "reporting"

    prediction_required = [
        "predictor.py",
        "signal_builder.py",
        "ranker.py",
        "__init__.py",
    ]

    reporting_required = [
        "prediction_writer.py",
        "ranked_reports.py",
        "metrics.py",
        "__init__.py",
    ]

    for file_name in prediction_required:
        assert (prediction_dir / file_name).exists(), f"Missing prediction file: {file_name}"

    for file_name in reporting_required:
        assert (reporting_dir / file_name).exists(), f"Missing reporting file: {file_name}"


def test_training_functions_extracted():
    root = Path(__file__).resolve().parents[1]
    training_dir = root / "src" / "gainify_stock_predictor" / "training"

    assert "def pretrain_bucket" in (training_dir / "yearly_pretrain.py").read_text(encoding="utf-8")
    assert "def run_yearly_pretrain" in (training_dir / "yearly_pretrain.py").read_text(encoding="utf-8")
    assert "def run_monthly_finetune" in (training_dir / "monthly_finetune.py").read_text(encoding="utf-8")
    assert "def run_weekly_finetune" in (training_dir / "weekly_finetune.py").read_text(encoding="utf-8")
    assert "def run_daily_finetune" in (training_dir / "daily_finetune.py").read_text(encoding="utf-8")

    losses_text = (training_dir / "losses.py").read_text(encoding="utf-8")
    assert "def spike_weighted_mse" in losses_text
    assert "def smooth_huber" in losses_text
    assert "def focal_bce_soft" in losses_text
    assert "LOSSES" in losses_text


def test_checkpoint_prediction_reporting_functions_extracted():
    root = Path(__file__).resolve().parents[1]

    checkpoint_dir = root / "src" / "gainify_stock_predictor" / "checkpoints"
    prediction_dir = root / "src" / "gainify_stock_predictor" / "prediction"
    reporting_dir = root / "src" / "gainify_stock_predictor" / "reporting"

    checkpoint_text = (checkpoint_dir / "checkpoint_manager.py").read_text(encoding="utf-8")
    metadata_text = (checkpoint_dir / "metadata_manager.py").read_text(encoding="utf-8")
    predictor_text = (prediction_dir / "predictor.py").read_text(encoding="utf-8")
    signal_text = (prediction_dir / "signal_builder.py").read_text(encoding="utf-8")
    writer_text = (reporting_dir / "prediction_writer.py").read_text(encoding="utf-8")
    ranked_text = (reporting_dir / "ranked_reports.py").read_text(encoding="utf-8")
    metrics_text = (reporting_dir / "metrics.py").read_text(encoding="utf-8")

    assert "def get_stage_output_dir" in checkpoint_text
    assert "def load_latest_successful_checkpoint" in checkpoint_text
    assert "def resolve_parent_checkpoint" in checkpoint_text
    assert "def save_stage_metadata" in metadata_text

    assert "def forecast_1d" in predictor_text
    assert "def run_predict_only" in predictor_text
    assert "def calculate_verdict" in signal_text
    assert "def ensemble_next_day_signal" in signal_text

    assert "def save_master_predictions_csv" in writer_text
    assert "def build_ranked_volatility_reports" in ranked_text
    assert "def evaluate_holdout_close" in metrics_text
'''

(PROJECT_ROOT / "tests" / "test_training_prediction_reporting_files.py").write_text(test_content, encoding="utf-8")

print("[OK] Wrote tests/test_training_prediction_reporting_files.py")
