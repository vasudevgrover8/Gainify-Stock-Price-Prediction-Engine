"""
Enhanced Stock Data Fetcher for NSE and BSE
Fetches complete historical data from listing day to present
Integrates market indicators with each stock record
Maintains local dataset with daily updates
FEATURES:
- Smart caching: Never re-downloads existing data
- Incremental updates: Only fetches new data since last update
- Automated daily updates: Scheduled updates for all stocks
- Progress tracking: Tracks which stocks have been processed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json
import yfinance as yf
from bs4 import BeautifulSoup
import schedule
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_fetcher.log'),
        logging.StreamHandler()
    ]
)

class EnhancedStockDataFetcher:
    def __init__(self, data_dir=None):
        base_dir = Path(__file__).resolve().parent
        if data_dir is None:
            self.data_dir = base_dir / 'stockno_data'
        else:
            self.data_dir = Path(data_dir).resolve()

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.stocks_file = self.data_dir / 'stocks_master.csv'
        self.historical_data_dir = self.data_dir / 'historical'
        self.historical_data_dir.mkdir(exist_ok=True)

        # Market indicators file (kept)
        self.market_indicators_file = self.data_dir / 'market_indicators_history.csv'

        # Progress tracking
        self.progress_file = self.data_dir / 'fetch_progress.json'
        self.last_update_file = self.data_dir / 'last_update.json'

        # NSE/BSE headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # Market indicators mapping
        # Core indicators (always used)
        self.core_indicators = {
            'USDINR': 'USDINR=X',
            'CRUDE_OIL': 'CL=F',
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
        }

        # Market-cap indices (ONLY ONE USED PER STOCK)
        self.marketcap_indices = {
            'NIFTY_SMALLCAP': 'BSE-SMLCAP.BO',
            'NIFTY_MIDCAP': 'NIFTYMIDCAP150.NS',
            'NIFTY_LARGECAP': '^CNX100',
        }

        # Sector indices (ONLY ONE USED PER STOCK)
        self.sector_indices = {
            'BANKNIFTY': '^NSEBANK',
            'NIFTY_PRIVATE_BANK': '^NIFTYPRBANK',
            'NIFTY_PSU_BANK': '^CNXPSUBANK',
            'NIFTY_FIN_SERVICE': '^CNXFIN',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_MIDSMALL_IT_TELECOM': '^NIFTYMSIT',
            'NIFTY_FMCG': '^CNXFMCG',
            'NIFTY_CONSUMER_DURABLES': '^CNXCONSUM',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_METAL': '^CNXMETAL',
            'NIFTY_CHEMICALS': '^NIFTYCHEM',
            'NIFTY_DEFENCE': '^NIFTYDEFENCE',
            'NIFTY_ENERGY': '^CNXENERGY',
            'NIFTY_OIL_GAS': '^NIFTYOILGAS',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'NIFTY_HEALTHCARE': '^NIFTYHEALTH',
            'NIFTY_MEDIA': '^CNXMEDIA',
            'NIFTY_REALTY': '^CNXREALTY',
        }

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

    def fetch_nse_stocks(self):
        """Fetch all NSE listed stocks"""
        logging.info("Fetching NSE stocks...")
        try:
            url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            session = requests.Session()
            session.headers.update(self.headers)
            session.get("https://www.nseindia.com", timeout=10)
            response = session.get(url, timeout=10)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                df['EXCHANGE'] = 'NSE'
                df['SYMBOL_EXCHANGE'] = df['SYMBOL'] + '.NS'
                logging.info(f"Fetched {len(df)} NSE stocks")
                return df
            else:
                return self._get_nse_from_yfinance()
        except Exception as e:
            logging.error(f"Error fetching NSE stocks: {e}")
            return self._get_nse_from_yfinance()

    def _get_nse_from_yfinance(self):
        """Backup method with expanded stock list"""
        logging.info("Using expanded stock list...")
        nse_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS',
            'KOTAKBANK.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
            'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
            'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS',
            'M&M.NS', 'TECHM.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'COALINDIA.NS',
            'BAJAJFINSV.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
            'BRITANNIA.NS', 'CIPLA.NS', 'SHRIRAMFIN.NS', 'TATACONSUM.NS', 'INDUSINDBK.NS',
            'APOLLOHOSP.NS', 'HINDALCO.NS', 'ADANIENT.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS',
            'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'BPCL.NS', 'IOC.NS', 'SIEMENS.NS'
        ]

        df = pd.DataFrame({
            'SYMBOL': [s.replace('.NS', '') for s in nse_stocks],
            'SYMBOL_EXCHANGE': nse_stocks,
            'EXCHANGE': 'NSE',
            'NAME': [s.replace('.NS', '') for s in nse_stocks]
        })
        return df

    def get_nifty50_stocks(self):
        """Get only Nifty 50 stocks"""
        nifty50_symbols = [
            'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
            'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS',
            'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
            'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
            'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
            'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS',
            'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
            'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS',
            'TATACONSUM.NS', 'TMPV.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS',
            'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'ETERNAL.NS',
        ]
        df = pd.DataFrame({
            'SYMBOL': [s.replace('.NS', '') for s in nifty50_symbols],
            'SYMBOL_EXCHANGE': nifty50_symbols,
            'EXCHANGE': 'NSE',
            'NAME': [s.replace('.NS', '') for s in nifty50_symbols]
        })
        logging.info(f"Nifty 50 stock list prepared: {len(df)} stocks")
        return df

    def fetch_bse_stocks(self):
        """Fetch BSE stocks"""
        logging.info("Fetching BSE stocks...")
        try:
            url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                df['EXCHANGE'] = 'BSE'
                df['SYMBOL_EXCHANGE'] = df['scrip_cd'].astype(str) + '.BO'
                logging.info(f"Fetched {len(df)} BSE stocks")
                return df
            else:
                logging.warning("Could not fetch from BSE API")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching BSE stocks: {e}")
            return pd.DataFrame()

    def get_all_stocks(self):
        """Get combined list of stocks"""
        nse_stocks = self.fetch_nse_stocks()
        time.sleep(2)
        # Uncomment below to include BSE
        # bse_stocks = self.fetch_bse_stocks()

        all_stocks = nse_stocks
        # if not bse_stocks.empty:
        #     all_stocks = pd.concat([nse_stocks, bse_stocks], ignore_index=True)

        all_stocks.to_csv(self.stocks_file, index=False)
        logging.info(f"Total stocks in master list: {len(all_stocks)}")
        return all_stocks

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


def main():
    """Main execution function"""
    fetcher = EnhancedStockDataFetcher()

    print("=" * 70)
    print("ENHANCED STOCK DATA FETCHER")
    print("=" * 70)
    print("\nFEATURES:")
    print("  ✓ Smart caching - Never re-downloads existing data")
    print("  ✓ Incremental updates - Only fetches new data since last update")
    print("  ✓ Automated daily updates - Scheduled updates for all stocks")
    print("  ✓ Progress tracking - Resumes from where it left off")
    print("\nThis will fetch COMPLETE HISTORICAL DATA for all stocks")
    print("from their listing day till today, integrated with:")
    print("  - NIFTY50, SENSEX, and stock-specific sector index (only)")
    print("  - Log transforms for those indices")
    print("  - Missing value handling (ffill + bfill)")
    print("  - Market cap in INR + close % change")
    print("  - Financial ratios (PE, PB, ROE, ROA, margins, etc.)")
    print("\nData will be saved to:")
    print(f"  - Individual stock files: {fetcher.historical_data_dir}/")
    print("=" * 70)

    # Check if data already exists
    progress = fetcher.load_progress()
    if progress['total_processed'] > 0:
        print(f"\n📊 Found existing data: {progress['total_processed']} stocks already processed")
        print(f"Last update: {progress.get('last_update', 'Unknown')}")
        print("\nOptions:")
        print("  1. Continue from where you left off (recommended)")
        print("  2. Update all stocks with latest data")
        print("  3. Start fresh (re-download everything)")
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == '3':
            force_update = True
            print("\n⚠️  Will re-download all data from scratch")
        elif choice == '2':
            force_update = False
            print("\n🔄 Will update all stocks with latest data")
        else:
            force_update = False
            print("\n▶️  Continuing from last position")
    else:
        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Exiting...")
            return
        force_update = False

    # Ask what stocks to download
    print("\nWhich stocks do you want to process?")
    print("  1. All NSE stocks (will take several hours)")
    print("  2. Nifty 50 stocks only (~50 stocks, fast)")
    stock_scope = input("Enter choice (1/2): ").strip()
    nifty50_only = (stock_scope == '2')

    if nifty50_only:
        print("\n📈 Nifty 50 mode selected - processing 50 stocks")
        max_stocks = None
    else:
        # Ask for number of stocks to process
        print("\nHow many stocks to process?")
        print("  - Enter a number (e.g., 10, 50)")
        print("  - Press Enter for ALL stocks (will take several hours)")
        max_stocks_input = input("Number of stocks: ").strip()
        max_stocks = None
        if max_stocks_input and max_stocks_input.isdigit():
            max_stocks = int(max_stocks_input)

    fetcher.fetch_and_save_all_stocks(max_stocks=max_stocks, force_update=force_update, nifty50_only=nifty50_only)

    print("\nDo you want to schedule automatic daily updates? (yes/no): ")
    response = input().lower()

    if response == 'yes':
        update_time = input("Enter update time (HH:MM, e.g., 09:30): ").strip()
        if not update_time:
            update_time = "09:30"

        print(f"\n✅ Scheduling daily updates at {update_time}...")
        print("The script will keep running. Press Ctrl+C to stop.")
        fetcher.schedule_daily_updates(update_time, max_stocks=max_stocks, nifty50_only=nifty50_only)
    else:
        print("\n✅ Data fetch complete! You can run this script again to update.")
        print("\nTo update daily automatically, run:")
        print("  python dataset.py")
        print("  and choose 'yes' for scheduled updates")


if __name__ == "__main__":
    main()