"""
Run yearly pretraining using modular pipeline.
"""

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


from gainify_stock_predictor.pipeline import run_yearly


if __name__ == "__main__":
    print("[START] Running modular yearly pretraining...", flush=True)
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}", flush=True)
    run_yearly()
    print("[DONE] Yearly pretraining finished.", flush=True)
