"""
Bucket configuration for Gainify Stock Predictor.

Moved from yearly.py without changing bucket definitions.
"""

IPO_RECENT_DAYS = 730


# ---------------------------------------------------------------
# Volatility-based bucket definitions
# ---------------------------------------------------------------
VOLATILITY_LEVELS = {
    "VERY_LOW": (0, 0.20),
    "LOW": (0.20, 0.30),
    "MEDIUM": (0.30, 0.40),
    "HIGH": (0.40, 0.60),
    "VERY_HIGH": (0.60, 1.5),
}


SECTORS = [
    "BANKING", "IT", "PHARMA", "ENERGY", "AUTO", "FMCG",
    "METAL", "INFRA", "TEXTILE", "CHEMICAL", "REALTY",
    "MEDIA", "TELECOM", "POWER", "CONSUMER_DURABLES",
]


BUCKETS = {}
bucket_id = 1

for vol_level in VOLATILITY_LEVELS.keys():
    for sector in SECTORS:
        BUCKETS[f"BUCKET_{bucket_id}"] = (vol_level, sector)
        bucket_id += 1

BUCKETS["BUCKET_76"] = ("VERY_HIGH", "GENERIC")
BUCKETS["BUCKET_77"] = ("VERY_LOW", "GENERIC")
BUCKETS["BUCKET_78"] = ("UNKNOWN", "UNKNOWN")

VOL_SECTOR_TO_BUCKET = {v: k for k, v in BUCKETS.items()}


# ---------------------------------------------------------------
# Sector keyword mapping
# ---------------------------------------------------------------
SECTOR_KEYWORD_MAP = {
    "bank": "BANKING",
    "financ": "BANKING",
    "nbfc": "BANKING",

    "information technology": "IT",
    "software": "IT",
    "tech": "IT",

    "pharma": "PHARMA",
    "hospital": "PHARMA",
    "health": "PHARMA",

    "oil": "ENERGY",
    "gas": "ENERGY",
    "energy": "ENERGY",
    "power": "ENERGY",

    "auto": "AUTO",
    "vehicle": "AUTO",
    "tyre": "AUTO",

    "fmcg": "FMCG",
    "consumer": "FMCG",
    "food": "FMCG",
    "beverage": "FMCG",

    "metal": "METAL",
    "steel": "METAL",
    "mining": "METAL",
    "alumin": "METAL",

    "infra": "INFRA",
    "construct": "INFRA",
    "cement": "INFRA",
    "road": "INFRA",

    "textile": "TEXTILE",
    "apparel": "TEXTILE",
    "garment": "TEXTILE",

    "chemical": "CHEMICAL",
    "fertilizer": "CHEMICAL",
    "pesticide": "CHEMICAL",

    "realty": "REALTY",
    "real estate": "REALTY",

    "media": "MEDIA",
    "entertainment": "MEDIA",
    "broadcast": "MEDIA",

    "telecom": "TELECOM",
    "communication": "TELECOM",

    "utility": "POWER",
    "electric": "POWER",

    "consumer durable": "CONSUMER_DURABLES",
    "appliance": "CONSUMER_DURABLES",
}