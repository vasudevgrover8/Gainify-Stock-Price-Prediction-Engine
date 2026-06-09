from datetime import datetime

import pandas as pd


def parse_date_safe(value):
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value)
    except (TypeError, ValueError):
        return None


def today_timestamp():
    return pd.Timestamp.today().normalize()


def format_date(value) -> str:
    parsed = parse_date_safe(value)
    if parsed is None:
        return ""
    if isinstance(parsed, datetime):
        return parsed.strftime("%Y-%m-%d")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")
