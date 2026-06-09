"""
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

def get_stage_output_dir(stage, bucket_tag, symbol=None, cutoff_date=None):
    """
    Returns the versioned output directory for a given stage/bucket/symbol.
    Structure:
        Models/<StageName>/<cutoff_date>/<bucket_tag>/[<symbol>/]
    """
    base = STAGE_DIRS.get(stage, os.path.join(MODEL_DIR, stage))
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    parts = [base, cutoff_str, bucket_tag]
    if symbol:
        parts.append(symbol)
    return os.path.join(*parts)


def load_latest_successful_checkpoint(stage, bucket_tag, symbol=None):
    """
    Scan the stage directory for the most recent checkpoint that has a
    valid stage_meta.json (indicating a successful completed write).
    Returns (checkpoint_path, meta_dict) or (None, None) if not found.

    Daily FT loads from weekly, weekly from monthly, monthly from yearly.
    This function only searches within its own stage's directory.
    """
    base = STAGE_DIRS.get(stage, os.path.join(MODEL_DIR, stage))
    if not os.path.isdir(base):
        return None, None

    candidates = []
    # Walk all cutoff_date subdirs, newest first
    for cutoff_dir in sorted(os.listdir(base), reverse=True):
        cutoff_path = os.path.join(base, cutoff_dir)
        if not os.path.isdir(cutoff_path):
            continue
        # Try bucket subdir then optional symbol subdir
        parts = [cutoff_path, bucket_tag]
        if symbol:
            parts.append(symbol)
        ckpt_dir = os.path.join(*parts)
        meta_path = os.path.join(ckpt_dir, "stage_meta.json")
        if os.path.isfile(meta_path):
            # Look for weights or SavedModel
            model_path = os.path.join(ckpt_dir, "model")
            weights_path = os.path.join(ckpt_dir, "best.weights.h5")
            enc_path = os.path.join(ckpt_dir, "encoder.weights.h5")
            if os.path.isdir(model_path):
                candidates.append((model_path, meta_path))
            elif os.path.isfile(weights_path):
                candidates.append((weights_path, meta_path))
            elif os.path.isfile(enc_path):
                candidates.append((enc_path, meta_path))

    if not candidates:
        return None, None

    ckpt_path, meta_path = candidates[0]
    with open(meta_path) as f:
        meta = json.load(f)
    log.info(f"[{stage}] Found checkpoint: {ckpt_path} (cutoff={meta.get('cutoff_date')})")
    return ckpt_path, meta


def resolve_parent_checkpoint(stage, bucket_tag, symbol=None):
    """
    Given the current stage, find the most recent successful checkpoint
    from the parent stage in the hierarchy.
    Hierarchy: yearly -> monthly -> weekly -> daily
    Returns (checkpoint_path, meta_dict).
    If no parent exists, returns (None, None).
    """
    parent_stage = STAGE_PARENT.get(stage)
    if not parent_stage:
        return None, None   # yearly has no parent
    ckpt_path, meta = load_latest_successful_checkpoint(parent_stage, bucket_tag, symbol=symbol)
    if ckpt_path:
        log.info(f"[{stage}] Resolved parent checkpoint from '{parent_stage}': {ckpt_path}")
    else:
        # Try bucket-level (encoder) if symbol-level not found
        ckpt_path, meta = load_latest_successful_checkpoint(parent_stage, bucket_tag, symbol=None)
        if ckpt_path:
            log.info(f"[{stage}] Resolved bucket-level parent from '{parent_stage}': {ckpt_path}")
    return ckpt_path, meta
