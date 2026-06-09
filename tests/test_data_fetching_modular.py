from pathlib import Path

import pytest


def test_data_fetching_files_exist():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "src" / "gainify_stock_predictor" / "data_fetching"

    required = [
        "data_fetcher.py",
        "progress_tracker.py",
        "stock_list_fetcher.py",
        "historical_fetcher.py",
        "market_indicator_fetcher.py",
        "data_integrator.py",
        "update_scheduler.py",
        "__init__.py",
    ]

    for file_name in required:
        assert (data_dir / file_name).exists(), f"Missing: {file_name}"


def test_data_fetcher_no_legacy_import():
    root = Path(__file__).resolve().parents[1]
    data_fetcher = root / "src" / "gainify_stock_predictor" / "data_fetching" / "data_fetcher.py"

    text = data_fetcher.read_text(encoding="utf-8")

    assert "legacy.DatasetComplete" not in text
    assert "from legacy" not in text
    assert "class EnhancedStockDataFetcher" in text


def test_data_fetching_methods_extracted():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "src" / "gainify_stock_predictor" / "data_fetching"

    assert "def save_progress" in (data_dir / "progress_tracker.py").read_text(encoding="utf-8")
    assert "def fetch_nse_stocks" in (data_dir / "stock_list_fetcher.py").read_text(encoding="utf-8")
    assert "def fetch_complete_historical_data" in (data_dir / "historical_fetcher.py").read_text(encoding="utf-8")
    assert "def fetch_market_indicators_history" in (data_dir / "market_indicator_fetcher.py").read_text(encoding="utf-8")
    assert "def integrate_stock_with_indicators" in (data_dir / "data_integrator.py").read_text(encoding="utf-8")
    assert "def fetch_and_save_all_stocks" in (data_dir / "update_scheduler.py").read_text(encoding="utf-8")


def test_modular_data_fetcher_imports():
    pytest.importorskip("yfinance")
    pytest.importorskip("schedule")

    from gainify_stock_predictor.data_fetching import EnhancedStockDataFetcher

    fetcher = EnhancedStockDataFetcher(data_dir="data/sample/test_fetcher")

    assert fetcher is not None
    assert hasattr(fetcher, "fetch_nse_stocks")
    assert hasattr(fetcher, "fetch_complete_historical_data")
    assert hasattr(fetcher, "fetch_market_indicators_history")
    assert hasattr(fetcher, "integrate_stock_with_indicators")
    assert hasattr(fetcher, "fetch_and_save_all_stocks")
