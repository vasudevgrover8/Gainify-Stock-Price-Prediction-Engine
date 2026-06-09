import pytest


def test_config_imports():
    import configs.paths_config
    import configs.model_config
    import configs.training_config
    import configs.bucket_config
    import configs.market_config


def test_data_fetcher_wrapper_import():
    pytest.importorskip("yfinance")
    pytest.importorskip("schedule")

    from gainify_stock_predictor.data_fetching import EnhancedStockDataFetcher

    assert EnhancedStockDataFetcher is not None


def test_legacy_yearly_path_exists():
    from gainify_stock_predictor.orchestrator_legacy import legacy_yearly_exists
    assert legacy_yearly_exists() is True
