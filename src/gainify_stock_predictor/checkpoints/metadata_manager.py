"""
Checkpoint metadata helpers.

Physically extracted from legacy/yearly.py.
Original metadata fields are preserved.
"""

import os
import json
from datetime import datetime

def save_stage_metadata(out_dir, stage, cutoff_date, window_start, window_end,
                        parent_checkpoint_path, bucket_tag, symbol=None, extra=None):
    """
    Save a JSON metadata file alongside a checkpoint.
    Fields:
        stage, cutoff_date, window_start, window_end,
        parent_checkpoint_path, bucket_tag, symbol, timestamp, ...extra
    """
    meta = {
        "stage":                   stage,
        "cutoff_date":             str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date),
        "window_start":            str(window_start) if window_start else None,
        "window_end":              str(window_end)   if window_end   else None,
        "parent_checkpoint_path":  str(parent_checkpoint_path) if parent_checkpoint_path else None,
        "bucket_tag":              bucket_tag,
        "symbol":                  symbol,
        "timestamp":               datetime.utcnow().isoformat(),
    }
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
