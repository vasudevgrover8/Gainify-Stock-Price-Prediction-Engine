"""
Market-indicator fetching methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


class MarketIndicatorFetcherMixin:
    """Methods extracted from legacy/DatasetComplete.py."""

    def fetch_market_indicators_history(self, start_date='1990-01-01', update_mode=False):
            """
            Fetch historical data for all market indicators
            If update_mode=True, only fetch recent data and merge with existing
            """
            logging.info("Fetching market indicators history...")

            all_indicators = {**self.core_indicators, **self.marketcap_indices, **self.sector_indices}

            # Load existing data if in update mode
            existing_data = None
            if update_mode and self.market_indicators_file.exists():
                existing_data = pd.read_csv(self.market_indicators_file)
                last_date = pd.to_datetime(existing_data['date'].max())
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                logging.info(f"Update mode: Fetching from {start_date}")

            market_data = None

            for name, ticker in all_indicators.items():
                try:
                    logging.info(f"Fetching {name} ({ticker})...")
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=datetime.now().strftime('%Y-%m-%d'),
                        progress=False,
                        group_by='column',
                        auto_adjust=False
                    )

                    if data is None or data.empty:
                        logging.warning(f"No new data for {name}")
                        continue

                    # Handle MultiIndex columns from yfinance safely
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    data.reset_index(inplace=True)

                    # yfinance sometimes returns 'Date' or 'Datetime'
                    if 'Date' in data.columns:
                        date_col = 'Date'
                    elif 'Datetime' in data.columns:
                        date_col = 'Datetime'
                    else:
                        date_col = data.columns[0]

                    data['date'] = pd.to_datetime(data[date_col]).dt.strftime('%Y-%m-%d')

                    # Ensure Close is a Series (not DataFrame)
                    close_series = data['Close']
                    if isinstance(close_series, pd.DataFrame):
                        close_series = close_series.iloc[:, -1]

                    data[name] = pd.to_numeric(close_series, errors='coerce')

                    if market_data is None:
                        market_data = data[['date', name]]
                    else:
                        market_data = market_data.merge(data[['date', name]], on='date', how='outer')

                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logging.error(f"Error fetching {name}: {e}")
                    continue

            if market_data is not None and not market_data.empty:
                if update_mode and existing_data is not None:
                    market_data = pd.concat([existing_data, market_data], ignore_index=True)
                    market_data = market_data.drop_duplicates(subset=['date'], keep='last')

                market_data = market_data.sort_values('date')

                # Forward fill missing values
                market_data = market_data.ffill().bfill()

                market_data.to_csv(self.market_indicators_file, index=False)
                logging.info(f"Saved market indicators: {len(market_data)} records")

                return market_data
            else:
                logging.error("Failed to fetch any market indicators")
                if existing_data is not None:
                    return existing_data
                return None

    def get_sector_index(self, sector, industry):
            """
            Map sector/industry to the most relevant sector index
            """
            sector = str(sector).lower().strip()
            industry = str(industry).lower().strip()

            if 'bank' in sector or 'bank' in industry:
                if 'private' in industry or 'private' in sector:
                    return 'NIFTY_PRIVATE_BANK'
                elif 'public' in industry or 'psu' in industry:
                    return 'NIFTY_PSU_BANK'
                else:
                    return 'BANKNIFTY'
            elif 'financial' in sector or 'finance' in industry:
                return 'NIFTY_FIN_SERVICE'

            elif 'technology' in sector or 'software' in industry or 'it services' in industry:
                return 'NIFTY_IT'
            elif 'telecom' in sector or 'telecom' in industry:
                return 'NIFTY_MIDSMALL_IT_TELECOM'

            elif 'consumer' in sector and 'durables' in sector:
                return 'NIFTY_CONSUMER_DURABLES'
            elif 'fmcg' in sector or 'consumer goods' in sector or 'food' in industry:
                return 'NIFTY_FMCG'

            elif 'auto' in sector or 'automobile' in industry:
                return 'NIFTY_AUTO'
            elif 'metal' in sector or 'steel' in industry or 'aluminium' in industry:
                return 'NIFTY_METAL'
            elif 'chemical' in sector or 'chemical' in industry:
                return 'NIFTY_CHEMICALS'
            elif 'defence' in sector or 'defence' in industry or 'aerospace' in industry:
                return 'NIFTY_DEFENCE'

            elif 'energy' in sector or 'power' in industry or 'utilities' in sector:
                return 'NIFTY_ENERGY'
            elif 'oil' in sector or 'gas' in sector or 'petroleum' in industry:
                return 'NIFTY_OIL_GAS'

            elif 'pharma' in sector or 'pharmaceutical' in industry:
                return 'NIFTY_PHARMA'
            elif 'health' in sector or 'hospital' in industry:
                return 'NIFTY_HEALTHCARE'

            elif 'media' in sector or 'entertainment' in industry:
                return 'NIFTY_MEDIA'
            elif 'real estate' in sector or 'realty' in sector:
                return 'NIFTY_REALTY'

            else:
                return 'NIFTY50'

    def get_marketcap_index(self, market_cap):
            """
            Determine market cap category based on market cap value
            Large Cap: > 20,000 Cr
            Mid Cap: 5,000 - 20,000 Cr
            Small Cap: < 5,000 Cr
            """
            try:
                if pd.isna(market_cap) or market_cap == 0:
                    return 'NIFTY_MIDCAP'

                market_cap_cr = market_cap / 10000000  # Convert to crores

                if market_cap_cr > 100000:
                    return 'NIFTY_LARGECAP'
                elif market_cap_cr > 50000:
                    return 'NIFTY_MIDCAP'
                else:
                    return 'NIFTY_SMALLCAP'
            except Exception:
                return 'NIFTY_MIDCAP'
