"""
Update existing stock dataset.

Uses your original DatasetComplete.py update_daily() logic.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


from gainify_stock_predictor.data_fetching import EnhancedStockDataFetcher


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--nifty50-only", action="store_true")

    args = parser.parse_args()

    fetcher = EnhancedStockDataFetcher(data_dir=args.data_dir)

    fetcher.update_daily(
        max_stocks=args.max_stocks,
        nifty50_only=args.nifty50_only,
    )


if __name__ == "__main__":
    main()
