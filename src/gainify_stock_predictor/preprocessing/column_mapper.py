"""
Column mapping utilities.

Constants are imported from configs.market_config.
No original mapping values are changed.
"""

from configs.market_config import COL_MAP, SECTOR_INDEX_INTERNAL


def standardize_columns(df):
    """
    Rename columns using COL_MAP.

    This helper is safe and optional.
    It does not remove any columns.
    """
    df = df.copy()
    rename_map = {}

    for col in df.columns:
        key = str(col).strip()
        lower_key = key.lower()

        if lower_key in COL_MAP:
            rename_map[col] = COL_MAP[lower_key]
        elif key in COL_MAP:
            rename_map[col] = COL_MAP[key]

    return df.rename(columns=rename_map)


def get_sector_index_columns():
    """
    Return sector index mapping exactly as configured.
    """
    return SECTOR_INDEX_INTERNAL