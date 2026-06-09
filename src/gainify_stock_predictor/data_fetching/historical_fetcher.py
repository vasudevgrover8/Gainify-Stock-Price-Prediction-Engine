"""
Historical stock-data fetching methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


class HistoricalFetcherMixin:
    """Methods extracted from legacy/DatasetComplete.py."""

    def fetch_complete_historical_data(self, symbol, start_date='1990-01-01', end_date=None):
            """Fetch complete historical data for a stock from listing day"""
            try:
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')

                ticker = yf.Ticker(symbol)

                # Get stock info (used for sector/industry/marketcap + financial ratios)
                info = ticker.info or {}

                # Fetch historical data
                hist = ticker.history(start=start_date, end=end_date)

                if hist.empty:
                    logging.warning(f"No historical data for {symbol}")
                    return None

                # Reset index to get date as column
                hist.reset_index(inplace=True)

                # Rename columns to lowercase
                hist.columns = hist.columns.str.lower()

                # Add metadata
                hist['symbol'] = symbol
                hist['date'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')

                # Extract key info
                try:
                    hist['sector'] = info.get('sector', 'Unknown')
                    hist['industry'] = info.get('industry', 'Unknown')
                    hist['market_cap'] = info.get('marketCap', 0)
                except Exception:
                    hist['sector'] = 'Unknown'
                    hist['industry'] = 'Unknown'
                    hist['market_cap'] = 0

                # -------------------------------
                # ✅ FINANCIAL RATIOS (ADDED BACK)
                # These are STATIC per stock (yfinance info snapshot),
                # repeated per row for ML convenience.
                # -------------------------------
                hist['pe_ratio'] = info.get('trailingPE', np.nan)
                hist['pb_ratio'] = info.get('priceToBook', np.nan)

                hist['roe'] = info.get('returnOnEquity', np.nan)
                hist['roa'] = info.get('returnOnAssets', np.nan)

                # debtToEquity can sometimes be percent-like; keep raw
                hist['debt_to_equity'] = info.get('debtToEquity', np.nan)

                hist['current_ratio'] = info.get('currentRatio', np.nan)
                hist['quick_ratio'] = info.get('quickRatio', np.nan)

                hist['profit_margins'] = info.get('profitMargins', np.nan)
                hist['operating_margins'] = info.get('operatingMargins', np.nan)
                hist['ebitda_margins'] = info.get('ebitdaMargins', np.nan)

                hist['payout_ratio'] = info.get('payoutRatio', np.nan)

                hist['revenue_growth'] = info.get('revenueGrowth', np.nan)

                # optional valuation multiples (safe)
                hist['enterprise_to_ebitda'] = info.get('enterpriseToEbitda', np.nan)
                hist['enterprise_to_revenue'] = info.get('enterpriseToRevenue', np.nan)

                logging.info(
                    f"Fetched {len(hist)} records for {symbol} "
                    f"from {hist['date'].min()} to {hist['date'].max()}"
                )
                return hist

            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                return None
