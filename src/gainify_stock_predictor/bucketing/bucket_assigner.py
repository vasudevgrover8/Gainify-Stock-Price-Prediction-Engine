"""
Bucket assignment logic.

Uses sector + volatility mappings from yearly.py.
"""

from configs.bucket_config import VOL_SECTOR_TO_BUCKET

from gainify_stock_predictor.bucketing.sector_mapper import map_sector_from_metadata
from gainify_stock_predictor.bucketing.volatility_buckets import (
    calculate_annualized_volatility,
    get_volatility_level,
    is_ipo_recent,
)


def assign_bucket(df, sector_str=None, industry_str=None):
    """
    Assign stock to one of the 78 buckets.

    Logic:
    - map sector/industry metadata to bucket sector
    - calculate annualized volatility
    - map volatility to volatility level
    - map (vol_level, sector) to bucket
    - fallback to generic/unknown buckets
    """
    if df is None or len(df) == 0:
        return "BUCKET_78"

    if sector_str is None and "sector" in df.columns:
        sector_str = df["sector"].iloc[-1]

    if industry_str is None and "industry" in df.columns:
        industry_str = df["industry"].iloc[-1]

    sector = map_sector_from_metadata(sector_str, industry_str)

    annual_vol = calculate_annualized_volatility(df)
    vol_level = get_volatility_level(annual_vol)

    bucket = VOL_SECTOR_TO_BUCKET.get((vol_level, sector))

    if bucket is not None:
        return bucket

    if sector == "GENERIC":
        if vol_level == "VERY_HIGH":
            return "BUCKET_76"
        elif vol_level == "VERY_LOW":
            return "BUCKET_77"

    return "BUCKET_78"
