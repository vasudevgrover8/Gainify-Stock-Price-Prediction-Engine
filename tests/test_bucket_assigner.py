import pandas as pd


def test_sector_mapper():
    from gainify_stock_predictor.bucketing import map_sector_from_metadata

    assert map_sector_from_metadata("Technology", "Software") == "IT"
    assert map_sector_from_metadata("Banking", "Financial Services") == "BANKING"


def test_volatility_level():
    from gainify_stock_predictor.bucketing import get_volatility_level

    assert get_volatility_level(0.10) == "VERY_LOW"
    assert get_volatility_level(0.25) == "LOW"
    assert get_volatility_level(0.35) == "MEDIUM"
    assert get_volatility_level(0.50) == "HIGH"
    assert get_volatility_level(0.80) == "VERY_HIGH"


def test_assign_bucket():
    from gainify_stock_predictor.bucketing import assign_bucket

    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=300),
        "Close": [100 + i * 0.1 for i in range(300)],
        "sector": ["Technology"] * 300,
        "industry": ["Software"] * 300,
    })

    bucket = assign_bucket(df)

    assert bucket.startswith("BUCKET_")
