"""
Batch update and scheduler methods.

Extracted from EnhancedStockDataFetcher in legacy/DatasetComplete.py.
"""

import logging
import time

import pandas as pd
import schedule


class UpdateSchedulerMixin:
    """Methods extracted from legacy/DatasetComplete.py."""

    def fetch_and_save_all_stocks(self, max_stocks=None, force_update=False, nifty50_only=False):
            """
            Fetch complete historical data for all stocks and integrate with indicators

            Args:
                max_stocks: Limit number of stocks to process
                force_update: If True, re-download all data. If False, only update new data
                nifty50_only: If True, only process Nifty 50 stocks
            """
            # Get stock list
            if nifty50_only:
                stocks_df = self.get_nifty50_stocks()
                logging.info("Mode: Nifty 50 stocks only")
            elif not self.stocks_file.exists():
                stocks_df = self.get_all_stocks()
            else:
                stocks_df = pd.read_csv(self.stocks_file)
                logging.info(f"Using existing stock list: {len(stocks_df)} stocks")

            if stocks_df.empty:
                logging.error("No stocks available")
                return

            # Fetch/Update market indicators
            if not self.market_indicators_file.exists() or force_update:
                logging.info("Fetching complete market indicators history...")
                market_data = self.fetch_market_indicators_history()
            else:
                logging.info("Updating market indicators...")
                market_data = self.fetch_market_indicators_history(update_mode=True)

            if market_data is None or market_data.empty:
                logging.error("Failed to get market indicators")
                return

            # Load progress
            progress = self.load_progress()
            processed_stocks = set(progress.get('processed_stocks', []))

            # Limit stocks if specified
            if max_stocks:
                stocks_df = stocks_df.head(max_stocks)

            newly_processed = []

            # Process each stock
            for idx, row in stocks_df.iterrows():
                symbol = row['SYMBOL_EXCHANGE']
                stock_file = self.historical_data_dir / f"{symbol.replace('.', '_')}.csv"

                # Skip if already processed and not forcing update
                if not force_update and symbol in processed_stocks and stock_file.exists():
                    logging.info(f"[{idx+1}/{len(stocks_df)}] Updating {symbol}")

                    last_date = self.get_last_date_in_stock_file(stock_file)

                    if last_date:
                        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')

                        if start_date < datetime.now().strftime('%Y-%m-%d'):
                            logging.info(f"Fetching new data from {start_date}")

                            stock_data = pd.read_csv(stock_file)
                            new_data = self.fetch_complete_historical_data(symbol, start_date=start_date)

                            if new_data is not None and not new_data.empty:
                                stock_data = pd.concat([stock_data, new_data], ignore_index=True)
                                stock_data = stock_data.drop_duplicates(subset=['date'], keep='last')
                                stock_data = stock_data.sort_values('date')

                                integrated = self.integrate_stock_with_indicators(stock_data, market_data)

                                if integrated is not None and not integrated.empty:
                                    integrated.to_csv(stock_file, index=False)
                                    logging.info(f"Updated {symbol}: Added {len(new_data)} new records")
                            else:
                                logging.info(f"No new data for {symbol}")
                        else:
                            logging.info(f"Already up to date: {symbol}")
                    else:
                        logging.warning(f"Could not determine last date for {symbol}, re-fetching")
                        stock_data = self.fetch_complete_historical_data(symbol)
                        if stock_data is not None and not stock_data.empty:
                            integrated = self.integrate_stock_with_indicators(stock_data, market_data)
                            if integrated is not None and not integrated.empty:
                                integrated.to_csv(stock_file, index=False)
                                newly_processed.append(symbol)
                else:
                    # First time processing this stock
                    logging.info(f"[{idx+1}/{len(stocks_df)}] Processing {symbol} (NEW)")

                    stock_data = self.fetch_complete_historical_data(symbol)

                    if stock_data is None or stock_data.empty:
                        logging.warning(f"Skipping {symbol} - no data available")
                        continue

                    integrated = self.integrate_stock_with_indicators(stock_data, market_data)

                    if integrated is not None and not integrated.empty:
                        integrated.to_csv(stock_file, index=False)
                        logging.info(f"Saved {len(integrated)} records for {symbol}")
                        newly_processed.append(symbol)

                # Rate limiting
                time.sleep(1)

                # Save progress every 10 new stocks
                if len(newly_processed) > 0 and len(newly_processed) % 10 == 0:
                    processed_stocks.update(newly_processed)
                    self.save_progress(list(processed_stocks))
                    logging.info(f"Progress saved: {len(processed_stocks)} stocks processed")

            # Final save
            if newly_processed:
                processed_stocks.update(newly_processed)
                self.save_progress(list(processed_stocks))

            self.save_last_update_time()
            logging.info(f"All stocks processed. Total: {len(processed_stocks)}, New: {len(newly_processed)}")

    def update_daily(self, max_stocks=None, nifty50_only=False):
            """Daily update - fetch latest data for all stocks"""
            logging.info("=" * 70)
            logging.info("DAILY UPDATE STARTED")
            logging.info("=" * 70)

            self.fetch_and_save_all_stocks(max_stocks=max_stocks, force_update=False, nifty50_only=nifty50_only)

            logging.info("=" * 70)
            logging.info("DAILY UPDATE COMPLETED")
            logging.info("=" * 70)

    def schedule_daily_updates(self, time_str="09:30", max_stocks=None, nifty50_only=False):
            """Schedule daily updates"""
            logging.info(f"Scheduling daily updates at {time_str}")
            schedule.every().day.at(time_str).do(self.update_daily, max_stocks=max_stocks, nifty50_only=nifty50_only)

            logging.info("Scheduler is running. Updates will occur automatically.")
            logging.info("Press Ctrl+C to stop.")

            while True:
                schedule.run_pending()
                time.sleep(60)
