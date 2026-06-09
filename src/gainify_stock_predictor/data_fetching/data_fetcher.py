"""
Main modular EnhancedStockDataFetcher class.

This no longer imports the old legacy dataset module.
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

    Core method bodies are extracted from the original dataset source file.
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
