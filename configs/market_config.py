"""
Market and column configuration.

Constants moved from:
- yearly.py
- DatasetComplete.py

No fetching/update logic changed.
"""

# ---------------------------------------------------------------
# CSV column mapping from yearly.py
# ---------------------------------------------------------------
COL_MAP = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "close_pct_change": "Change %",
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
}


SECTOR_INDEX_INTERNAL = {
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_MEDIA": "^CNXMEDIA",
    "NIFTY_MIDCAP100": "^NIFTY_MIDCAP_100",
    "NIFTY_SMLCAP100": "BSE-SMLCAP.BO",
    "sector_index_value": "sector_index_value",
}


# ---------------------------------------------------------------
# DatasetComplete.py request headers
# ---------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------
# DatasetComplete.py market indicator mappings
# ---------------------------------------------------------------
CORE_INDICATORS = {
    "USDINR": "USDINR=X",
    "CRUDE_OIL": "CL=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
}


MARKETCAP_INDICES = {
    "NIFTY_SMALLCAP": "BSE-SMLCAP.BO",
    "NIFTY_MIDCAP": "NIFTYMIDCAP150.NS",
    "NIFTY_LARGECAP": "^CNX100",
}


SECTOR_INDICES = {
    "BANKNIFTY": "^NSEBANK",
    "NIFTY_PRIVATE_BANK": "^NIFTYPRBANK",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
    "NIFTY_FIN_SERVICE": "^CNXFIN",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_MIDSMALL_IT_TELECOM": "^NIFTYMSIT",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_CONSUMER_DURABLES": "^CNXCONSUM",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_CHEMICALS": "^NIFTYCHEM",
    "NIFTY_DEFENCE": "^NIFTYDEFENCE",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_OIL_GAS": "^NIFTYOILGAS",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_HEALTHCARE": "^NIFTYHEALTH",
    "NIFTY_MEDIA": "^CNXMEDIA",
    "NIFTY_REALTY": "^CNXREALTY",
}