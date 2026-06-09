"""
Run full modular pipeline.

Stages:
1. yearly_pretrain
2. monthly_finetune
3. weekly_finetune
4. daily_finetune
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


from gainify_stock_predictor.pipeline.stage_orchestrator import run_stage


if __name__ == "__main__":
    run_stage("full_pipeline")
