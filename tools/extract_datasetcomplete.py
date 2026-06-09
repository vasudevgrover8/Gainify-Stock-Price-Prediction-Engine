"""
Extract EnhancedStockDataFetcher methods from legacy/DatasetComplete.py
into modular data_fetching files.

This does not rewrite logic.
It copies exact method bodies from the original class.
"""

import ast
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATASET = PROJECT_ROOT / "legacy" / "DatasetComplete.py"
DATA_FETCHING_DIR = PROJECT_ROOT / "src" / "gainify_stock_predictor" / "data_fetching"


source = LEGACY_DATASET.read_text(encoding="utf-8")
tree = ast.parse(source)


def get_class_node(class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise RuntimeError(f"Class not found: {class_name}")


FETCHER_CLASS = get_class_node("EnhancedStockDataFetcher")


def extract_method(method_name):
    for node in FETCHER_CLASS.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise RuntimeError(f"Could not extract method: {method_name}")
            return textwrap.dedent(text).rstrip()
    raise RuntimeError(f"Method not found: {method_name}")


def indent_method(text):
    return textwrap.indent(text, "    ")


def extract_methods(method_names):
    blocks = []
    missing = []

    for name in method_names:
        try:
            blocks.append(indent_method(extract_method(name)))
        except RuntimeError:
            missing.append(name)

    if missing:
        print("[WARN] Missing methods:", missing)

    return "\n\n".join(blocks)


def write_mixin(path, header, class_name, method_names):
    body = extract_methods(method_names)

    content = header.rstrip() + "\n\n\n"
    content += f"class {class_name}:\n"
    content += f'    """Methods extracted from legacy/DatasetComplete.py."""\n\n'

    if body.strip():
        content += body + "\n"
    else:
        content += "    pass\n"

    path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {path.relative_to(PROJECT_ROOT)}")


write_mixin(
    DATA_FETCHING_DIR / "progress_tracker.py",
    '''"""
Progress tracking methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import json
import logging

import pandas as pd
''',
    "ProgressTrackerMixin",
    [
        "save_progress",
        "load_progress",
        "save_last_update_time",
        "get_last_update_date",
        "get_last_date_in_stock_file",
    ],
)


write_mixin(
    DATA_FETCHING_DIR / "stock_list_fetcher.py",
    '''"""
Stock-list fetching methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
import time
from io import StringIO

import pandas as pd
import requests
''',
    "StockListFetcherMixin",
    [
        "fetch_nse_stocks",
        "_get_nse_from_yfinance",
        "get_nifty50_stocks",
        "fetch_bse_stocks",
        "get_all_stocks",
    ],
)


write_mixin(
    DATA_FETCHING_DIR / "historical_fetcher.py",
    '''"""
Historical stock-data fetching methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
''',
    "HistoricalFetcherMixin",
    [
        "fetch_complete_historical_data",
    ],
)


write_mixin(
    DATA_FETCHING_DIR / "market_indicator_fetcher.py",
    '''"""
Market-indicator fetching methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
''',
    "MarketIndicatorFetcherMixin",
    [
        "fetch_market_indicators_history",
        "get_sector_index",
        "get_marketcap_index",
    ],
)


write_mixin(
    DATA_FETCHING_DIR / "data_integrator.py",
    '''"""
Stock + market-indicator integration methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging

import numpy as np
import pandas as pd
''',
    "DataIntegratorMixin",
    [
        "integrate_stock_with_indicators",
    ],
)


write_mixin(
    DATA_FETCHING_DIR / "update_scheduler.py",
    '''"""
Batch update and scheduler methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
import time

import pandas as pd
import schedule
''',
    "UpdateSchedulerMixin",
    [
        "fetch_and_save_all_stocks",
        "update_daily",
        "schedule_daily_updates",
    ],
)


data_fetcher_content = '''"""
Main modular EnhancedStockDataFetcher class.

This no longer imports legacy.DatasetComplete.
Method groups are inherited from modular mixins.
"""

from pathlib import Path

from configs.market_config import (
    HEADERS,
    CORE_INDICATORS,
    MARKETCAP_INDICES,
    SECTOR_INDICES,
)

from gainify_stock_predictor.data_fetching.progress_tracker import ProgressTrackerMixin
from gainify_stock_predictor.data_fetching.stock_list_fetcher import StockListFetcherMixin
from gainify_stock_predictor.data_fetching.historical_fetcher import HistoricalFetcherMixin
from gainify_stock_predictor.data_fetching.market_indicator_fetcher import MarketIndicatorFetcherMixin
from gainify_stock_predictor.data_fetching.data_integrator import DataIntegratorMixin
from gainify_stock_predictor.data_fetching.update_scheduler import UpdateSchedulerMixin


class EnhancedStockDataFetcher(
    ProgressTrackerMixin,
    StockListFetcherMixin,
    HistoricalFetcherMixin,
    MarketIndicatorFetcherMixin,
    DataIntegratorMixin,
    UpdateSchedulerMixin,
):
    """
    Modular version of the original EnhancedStockDataFetcher.

    Core method bodies are extracted from legacy/DatasetComplete.py.
    """

    def __init__(self, data_dir=None):
        base_dir = Path(__file__).resolve().parent

        if data_dir is None:
            self.data_dir = base_dir / "stockno_data"
        else:
            self.data_dir = Path(data_dir).resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.stocks_file = self.data_dir / "stocks_master.csv"
        self.historical_data_dir = self.data_dir / "historical"
        self.historical_data_dir.mkdir(exist_ok=True)

        # Market indicators file
        self.market_indicators_file = self.data_dir / "market_indicators_history.csv"

        # Progress tracking
        self.progress_file = self.data_dir / "fetch_progress.json"
        self.last_update_file = self.data_dir / "last_update.json"

        # NSE/BSE headers
        self.headers = HEADERS

        # Market indicators mapping
        self.core_indicators = CORE_INDICATORS
        self.marketcap_indices = MARKETCAP_INDICES
        self.sector_indices = SECTOR_INDICES
'''

(DATA_FETCHING_DIR / "data_fetcher.py").write_text(data_fetcher_content, encoding="utf-8")
print("[OK] Wrote src/gainify_stock_predictor/data_fetching/data_fetcher.py")


init_content = '''from gainify_stock_predictor.data_fetching.data_fetcher import EnhancedStockDataFetcher

__all__ = ["EnhancedStockDataFetcher"]
'''

(DATA_FETCHING_DIR / "__init__.py").write_text(init_content, encoding="utf-8")
print("[OK] Updated data_fetching/__init__.py")
