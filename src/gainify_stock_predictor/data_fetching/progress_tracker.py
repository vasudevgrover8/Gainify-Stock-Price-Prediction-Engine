"""
Progress tracking methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import json
import logging

import pandas as pd


class ProgressTrackerMixin:
    """Methods extracted from legacy/DatasetComplete.py."""

    def save_progress(self, processed_stocks):
            """Save progress to JSON file"""
            progress_data = {
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processed_stocks': processed_stocks,
                'total_processed': len(processed_stocks)
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)

    def load_progress(self):
            """Load progress from JSON file"""
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            return {'processed_stocks': [], 'total_processed': 0}

    def save_last_update_time(self):
            """Save the last update timestamp"""
            update_data = {
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_update_date': datetime.now().strftime('%Y-%m-%d')
            }
            with open(self.last_update_file, 'w') as f:
                json.dump(update_data, f, indent=2)

    def get_last_update_date(self):
            """Get the last update date"""
            if self.last_update_file.exists():
                with open(self.last_update_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_update_date')
            return None

    def get_last_date_in_stock_file(self, stock_file):
            """Get the last date available in a stock file"""
            try:
                if stock_file.exists():
                    df = pd.read_csv(stock_file)
                    if not df.empty and 'date' in df.columns:
                        return pd.to_datetime(df['date'].max()).strftime('%Y-%m-%d')
            except Exception as e:
                logging.error(f"Error reading last date from {stock_file}: {e}")
            return None
