"""
Stock + market-indicator integration methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging

import numpy as np
import pandas as pd


class DataIntegratorMixin:
    """Methods extracted from legacy/DatasetComplete.py."""

    def integrate_stock_with_indicators(self, stock_data, market_data):
            """
            Integrate stock data with market indicators
            Only includes relevant sector index + NIFTY50 + SENSEX (no extra indices)
            Adds:
              - missing value fill for the included indices (ffill+bfill)
              - log transforms for NIFTY50, SENSEX, and stock-specific sector index
              - marketcap in INR (market_cap_inr)
              - daily % change of close (close_pct_change)
            """
            if stock_data is None or stock_data.empty:
                return None

            if market_data is None or market_data.empty:
                logging.warning("No market data available for integration")
                return stock_data

            # Get sector info
            sector = stock_data['sector'].iloc[0] if 'sector' in stock_data.columns else 'Unknown'
            industry = stock_data['industry'].iloc[0] if 'industry' in stock_data.columns else 'Unknown'
            sector_index = self.get_sector_index(sector, industry)

            # Select ONLY relevant indicators
            relevant_cols = ['date']

            for col in ['NIFTY50', 'SENSEX']:
                if col in market_data.columns:
                    relevant_cols.append(col)

            if sector_index in market_data.columns:
                relevant_cols.append(sector_index)

            relevant_cols = list(dict.fromkeys(relevant_cols))
            market_subset = market_data[relevant_cols].copy()

            # Merge cleanly
            integrated = stock_data.merge(market_subset, on='date', how='left')

            # Remove duplicated columns if any (safety)
            integrated = integrated.loc[:, ~integrated.columns.duplicated(keep='last')]

            # Fill missing values for included index columns
            for col in relevant_cols:
                if col != 'date' and col in integrated.columns:
                    s = integrated[col]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, -1]
                    integrated[col] = pd.to_numeric(s, errors='coerce').ffill().bfill()

            # Log transforms (only indices you asked)
            log_targets = ['NIFTY50', 'SENSEX', sector_index]
            for c in log_targets:
                if c in integrated.columns:
                    s = integrated[c]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, -1]
                    s = pd.to_numeric(s, errors='coerce')
                    integrated[f'log_{c}'] = np.log(s.replace(0, np.nan))
                    integrated[f'log_{c}'] = integrated[f'log_{c}'].ffill().bfill()

            # Keep your standardized output columns (do not change logic)
            integrated['sector_index_name'] = sector_index
            if sector_index in integrated.columns:
                integrated['sector_index_value'] = integrated[sector_index]
            elif 'NIFTY50' in integrated.columns:
                integrated['sector_index_value'] = integrated['NIFTY50']
            else:
                integrated['sector_index_value'] = np.nan

            # Market cap in INR (requested)
            integrated['market_cap_inr'] = integrated['market_cap'] if 'market_cap' in integrated.columns else np.nan

            # Daily % change feature (requested)
            if 'close' in integrated.columns:
                integrated['close_pct_change'] = pd.to_numeric(integrated['close'], errors='coerce').pct_change() * 100
                integrated['close_pct_change'] = integrated['close_pct_change'].fillna(0.0)

            return integrated
