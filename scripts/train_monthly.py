"""
Run monthly fine-tuning using modular pipeline.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


from gainify_stock_predictor.pipeline import run_monthly


if __name__ == "__main__":
    run_monthly()
