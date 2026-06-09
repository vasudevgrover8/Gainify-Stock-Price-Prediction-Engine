"""
Sector mapping logic.

Moved from yearly.py.
"""

from configs.bucket_config import SECTOR_KEYWORD_MAP


def map_sector_from_metadata(sector_str, industry_str):
    combined = (str(sector_str) + " " + str(industry_str)).lower()

    for kw, bucket_sec in SECTOR_KEYWORD_MAP.items():
        if kw in combined:
            return bucket_sec

    return "GENERIC"
