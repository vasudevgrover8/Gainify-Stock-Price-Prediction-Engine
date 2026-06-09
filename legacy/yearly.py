# ================================================================
# Orchestrator v3: Staged Hierarchical Transfer Learning for NSE Stocks
# CSV-based pipeline | 78 Buckets (Volatility × Sector) | CNN+TFT+Multi-Head Attention
# 4-tier training: Yearly Pretrain -> Monthly FT -> Weekly FT -> Daily FT
# Auto cutoff-date detection | Checkpoint hierarchy | Master CSV outputs
# No visualization outputs in main run path
# ================================================================

import os, re, warnings, random, pickle, glob, argparse, json, logging
from datetime import datetime, date
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; prevents any GUI pop-ups
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pandas.tseries.offsets import BDay
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
import joblib
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------
# ===== STAGE CONTROL ===========================================
# How to run different stages:
#   Set RUN_STAGE to one of:
#       "yearly_pretrain"   – full history pretrain per bucket
#       "monthly_finetune"  – loads yearly checkpoint, trains on MONTHLY_LOOKBACK_DAYS
#       "weekly_finetune"   – loads monthly checkpoint, trains on WEEKLY_LOOKBACK_DAYS
#       "daily_finetune"    – loads weekly checkpoint, trains on DAILY_LOOKBACK_DAYS
#       "predict_only"      – loads best available checkpoint, runs forecast only
#       "full_pipeline"     – runs all four stages sequentially
#
# AUTO_DETECT_CUTOFF_DATE: when True, the cutoff date for any stage
#   is automatically set to the latest date currently present in the dataset.
#   No manual entry needed. Appending new rows daily automatically shifts
#   the cutoff when that stage is next run explicitly.
#
# AUTO_RESOLVE_PARENT_CHECKPOINT: when True, each stage automatically
#   locates the most recent successful parent-stage checkpoint.
#
# ENABLE_VISUALS: set False to suppress all matplotlib chart outputs
#   (default False for production runs; set True for ad-hoc inspection).
# ---------------------------------------------------------------
RUN_STAGE                  = "daily_finetune"   # <-- change this to control which stage runs
AUTO_DETECT_CUTOFF_DATE    = True
AUTO_RESOLVE_PARENT_CHECKPOINT = True
ENABLE_VISUALS             = False              # True to re-enable charts for debugging

# Stage-specific rolling lookback windows (calendar days)
MONTHLY_LOOKBACK_DAYS  = 365    # ~12 months of data for monthly FT
WEEKLY_LOOKBACK_DAYS   = 180    # ~6 months of data for weekly FT
DAILY_LOOKBACK_DAYS    = 90     # ~3 months of data for daily FT

# ---------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR    = r"C:\Stock Price Predictor"
DATA_DIR    = os.path.join(BASE_DIR, "Stock_Data", "historical")
MODEL_DIR   = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Outputs")

# Stage checkpoint directories (parent-child hierarchy)
# Yearly -> Monthly -> Weekly -> Daily
STAGE_DIRS = {
    "yearly_pretrain":  os.path.join(MODEL_DIR, "Yearly_Pretrained"),
    "monthly_finetune": os.path.join(MODEL_DIR, "Monthly_Finetuned"),
    "weekly_finetune":  os.path.join(MODEL_DIR, "Weekly_Finetuned"),
    "daily_finetune":   os.path.join(MODEL_DIR, "Daily_Finetuned"),
}
STAGE_PARENT = {
    "monthly_finetune": "yearly_pretrain",
    "weekly_finetune":  "monthly_finetune",
    "daily_finetune":   "weekly_finetune",
}

# ---------------------------------------------------------------
# Training Knobs
# ---------------------------------------------------------------
SEQ_LEN              = 90
EPOCHS_PT            = 35
EPOCHS_MONTHLY_FT    = 25
EPOCHS_WEEKLY_FT     = 20
EPOCHS_DAILY_FT      = 5
BATCH                = 32
LR_PT                = 8e-4
LR_MONTHLY_FT        = 3e-4
LR_WEEKLY_FT         = 2e-4
LR_DAILY_FT          = 5e-5
L2REG                = 1e-6

EMBARGO_STEPS           = 20
DROPOUT_ENC_PRETRAIN    = 0.35
DROPOUT_ENC_FINETUNE    = 0.20
DROPOUT_HEAD            = 0.25
FREEZE_ENCODER_LAYERS   = False
LABEL_SMOOTH_EPS        = 0.01

IPO_RECENT_DAYS  = 730
MIN_HISTORY_BARS = 60

# ---------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "orchestrator.log"), mode="a")
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Volatility-based bucket definitions
# ---------------------------------------------------------------
VOLATILITY_LEVELS = {
    "VERY_LOW":  (0,    0.20),
    "LOW":       (0.20, 0.30),
    "MEDIUM":    (0.30, 0.40),
    "HIGH":      (0.40, 0.60),
    "VERY_HIGH": (0.60, 1.5)
}

SECTORS = [
    "BANKING", "IT", "PHARMA", "ENERGY", "AUTO", "FMCG",
    "METAL", "INFRA", "TEXTILE", "CHEMICAL", "REALTY",
    "MEDIA", "TELECOM", "POWER", "CONSUMER_DURABLES"
]

BUCKETS = {}
bucket_id = 1
for vol_level in VOLATILITY_LEVELS.keys():
    for sector in SECTORS:
        BUCKETS[f"BUCKET_{bucket_id}"] = (vol_level, sector)
        bucket_id += 1

BUCKETS["BUCKET_76"] = ("VERY_HIGH", "GENERIC")
BUCKETS["BUCKET_77"] = ("VERY_LOW",  "GENERIC")
BUCKETS["BUCKET_78"] = ("UNKNOWN",   "UNKNOWN")

VOL_SECTOR_TO_BUCKET = {v: k for k, v in BUCKETS.items()}

# ---------------------------------------------------------------
# CSV column mapping
# ---------------------------------------------------------------
COL_MAP = {
    "date":             "Date",
    "open":             "Open",
    "high":             "High",
    "low":              "Low",
    "close":            "Close",
    "volume":           "Volume",
    "close_pct_change": "Change %",
    "NIFTY50":          "^NSEI",
    "SENSEX":           "^BSESN",
}

SECTOR_INDEX_INTERNAL = {
    "NIFTY_IT":          "^CNXIT",
    "NIFTY_BANK":        "^NSEBANK",
    "NIFTY_AUTO":        "^CNXAUTO",
    "NIFTY_FMCG":        "^CNXFMCG",
    "NIFTY_PHARMA":      "^CNXPHARMA",
    "NIFTY_METAL":       "^CNXMETAL",
    "NIFTY_ENERGY":      "^CNXENERGY",
    "NIFTY_REALTY":      "^CNXREALTY",
    "NIFTY_INFRA":       "^CNXINFRA",
    "NIFTY_MEDIA":       "^CNXMEDIA",
    "NIFTY_MIDCAP100":   "^NIFTY_MIDCAP_100",
    "NIFTY_SMLCAP100":   "BSE-SMLCAP.BO",
    "sector_index_value": "sector_index_value",
}

# ---------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------
SECTOR_KEYWORD_MAP = {
    "bank": "BANKING", "financ": "BANKING", "nbfc": "BANKING",
    "information technology": "IT", "software": "IT", "tech": "IT",
    "pharma": "PHARMA", "hospital": "PHARMA", "health": "PHARMA",
    "oil": "ENERGY", "gas": "ENERGY", "energy": "ENERGY", "power": "ENERGY",
    "auto": "AUTO", "vehicle": "AUTO", "tyre": "AUTO",
    "fmcg": "FMCG", "consumer": "FMCG", "food": "FMCG", "beverage": "FMCG",
    "metal": "METAL", "steel": "METAL", "mining": "METAL", "alumin": "METAL",
    "infra": "INFRA", "construct": "INFRA", "cement": "INFRA", "road": "INFRA",
    "textile": "TEXTILE", "apparel": "TEXTILE", "garment": "TEXTILE",
    "chemical": "CHEMICAL", "fertilizer": "CHEMICAL", "pesticide": "CHEMICAL",
    "realty": "REALTY", "real estate": "REALTY",
    "media": "MEDIA", "entertainment": "MEDIA", "broadcast": "MEDIA",
    "telecom": "TELECOM", "communication": "TELECOM",
    "utility": "POWER", "electric": "POWER",
    "consumer durable": "CONSUMER_DURABLES", "appliance": "CONSUMER_DURABLES"
}

def map_sector_from_metadata(sector_str, industry_str):
    combined = (str(sector_str) + " " + str(industry_str)).lower()
    for kw, bucket_sec in SECTOR_KEYWORD_MAP.items():
        if kw in combined:
            return bucket_sec
    return "GENERIC"

# ---------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------
def calculate_annualized_volatility(df, window=252):
    if 'LogRet' not in df.columns:
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    log_rets = df['LogRet'].dropna()
    if len(log_rets) < 20:
        return 0.3
    lookback    = min(window, len(log_rets))
    recent_rets = log_rets.iloc[-lookback:]
    daily_vol   = recent_rets.std()
    annual_vol  = daily_vol * np.sqrt(252)
    return np.clip(annual_vol, 0.05, 1.5)

def get_volatility_level(annual_vol):
    if annual_vol < 0.20:   return "VERY_LOW"
    elif annual_vol < 0.30: return "LOW"
    elif annual_vol < 0.40: return "MEDIUM"
    elif annual_vol < 0.60: return "HIGH"
    else:                   return "VERY_HIGH"

def is_ipo_recent(df, threshold_days=IPO_RECENT_DAYS):
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(dates) < 2:
            return True
        listing_date = dates.min()
        today        = pd.Timestamp.today().normalize()
        return (today - listing_date).days < threshold_days
    except Exception:
        return False

# ---------------------------------------------------------------
# ===== AUTO CUTOFF-DATE HELPERS ================================
# These helpers implement auto-detection of the latest available
# date in the dataset, so no manual date entry is needed.
# Each stage calls detect_latest_dataset_date() at startup and
# uses that as its run_cutoff_date.
# ---------------------------------------------------------------

def parse_date_column(df):
    """Return a sorted Series of parsed dates from any recognisable date column."""
    for col in ["Date", "date", "DATE", "Timestamp", "timestamp"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            return parsed.sort_values()
    return pd.Series(dtype="datetime64[ns]")

def detect_latest_dataset_date(data_dir=DATA_DIR):
    """
    Scan all CSV files in data_dir, find the globally latest date present.
    This becomes the run_cutoff_date for whichever stage is currently executing.
    Appending new rows to CSVs automatically shifts this date forward next time
    that stage is explicitly run.
    """
    latest = None
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        log.warning(f"No CSV files found in {data_dir} for cutoff-date detection.")
        return pd.Timestamp.today().normalize()
    for path in csv_files:
        try:
            # Read only the date column to keep memory low
            df_peek = pd.read_csv(path, usecols=lambda c: c.strip().lower() in
                                  ["date", "timestamp"], low_memory=False, nrows=None)
            df_peek.columns = [c.strip() for c in df_peek.columns]
            dates = parse_date_column(df_peek)
            if len(dates) > 0:
                file_latest = dates.max()
                if latest is None or file_latest > latest:
                    latest = file_latest
        except Exception:
            pass
    if latest is None:
        latest = pd.Timestamp.today().normalize()
    log.info(f"Auto-detected dataset latest date: {latest.date()}")
    return latest

def filter_df_to_cutoff(df, cutoff_date):
    """Return df rows with Date <= cutoff_date. Preserves all columns."""
    if "Date" not in df.columns:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    cutoff_ts  = pd.Timestamp(cutoff_date)
    return df[df["Date"] <= cutoff_ts].reset_index(drop=True)

def apply_stage_window(df, stage, cutoff_date):
    """
    Filter df to the rolling lookback window for the given stage.
    - yearly_pretrain: full history up to cutoff_date
    - monthly_finetune: MONTHLY_LOOKBACK_DAYS before cutoff_date
    - weekly_finetune:  WEEKLY_LOOKBACK_DAYS before cutoff_date
    - daily_finetune:   DAILY_LOOKBACK_DAYS before cutoff_date
    """
    df = filter_df_to_cutoff(df, cutoff_date)
    cutoff_ts = pd.Timestamp(cutoff_date)
    if stage == "yearly_pretrain":
        return df          # use all history
    lookback_map = {
        "monthly_finetune": MONTHLY_LOOKBACK_DAYS,
        "weekly_finetune":  WEEKLY_LOOKBACK_DAYS,
        "daily_finetune":   DAILY_LOOKBACK_DAYS,
    }
    days = lookback_map.get(stage, DAILY_LOOKBACK_DAYS)
    window_start = cutoff_ts - pd.Timedelta(days=days)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"] >= window_start].reset_index(drop=True)
    return df

# ---------------------------------------------------------------
# ===== CHECKPOINT HELPERS ======================================
# Each stage saves a metadata JSON alongside weights.
# load_latest_successful_checkpoint() scans the stage directory
# for the most recent checkpoint that has a valid metadata file,
# preventing half-written checkpoints from being loaded.
# resolve_parent_checkpoint() walks up the stage hierarchy to
# find the correct parent weights when starting a new stage.
# ---------------------------------------------------------------

def get_stage_output_dir(stage, bucket_tag, symbol=None, cutoff_date=None):
    """
    Returns the versioned output directory for a given stage/bucket/symbol.
    Structure:
        Models/<StageName>/<cutoff_date>/<bucket_tag>/[<symbol>/]
    """
    base = STAGE_DIRS.get(stage, os.path.join(MODEL_DIR, stage))
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    parts = [base, cutoff_str, bucket_tag]
    if symbol:
        parts.append(symbol)
    return os.path.join(*parts)

def save_stage_metadata(out_dir, stage, cutoff_date, window_start, window_end,
                        parent_checkpoint_path, bucket_tag, symbol=None, extra=None):
    """
    Save a JSON metadata file alongside a checkpoint.
    Fields:
        stage, cutoff_date, window_start, window_end,
        parent_checkpoint_path, bucket_tag, symbol, timestamp, ...extra
    """
    meta = {
        "stage":                   stage,
        "cutoff_date":             str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date),
        "window_start":            str(window_start) if window_start else None,
        "window_end":              str(window_end)   if window_end   else None,
        "parent_checkpoint_path":  str(parent_checkpoint_path) if parent_checkpoint_path else None,
        "bucket_tag":              bucket_tag,
        "symbol":                  symbol,
        "timestamp":               datetime.utcnow().isoformat(),
    }
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stage_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path

def load_latest_successful_checkpoint(stage, bucket_tag, symbol=None):
    """
    Scan the stage directory for the most recent checkpoint that has a
    valid stage_meta.json (indicating a successful completed write).
    Returns (checkpoint_path, meta_dict) or (None, None) if not found.

    Daily FT loads from weekly, weekly from monthly, monthly from yearly.
    This function only searches within its own stage's directory.
    """
    base = STAGE_DIRS.get(stage, os.path.join(MODEL_DIR, stage))
    if not os.path.isdir(base):
        return None, None

    candidates = []
    # Walk all cutoff_date subdirs, newest first
    for cutoff_dir in sorted(os.listdir(base), reverse=True):
        cutoff_path = os.path.join(base, cutoff_dir)
        if not os.path.isdir(cutoff_path):
            continue
        # Try bucket subdir then optional symbol subdir
        parts = [cutoff_path, bucket_tag]
        if symbol:
            parts.append(symbol)
        ckpt_dir = os.path.join(*parts)
        meta_path = os.path.join(ckpt_dir, "stage_meta.json")
        if os.path.isfile(meta_path):
            # Look for weights or SavedModel
            model_path = os.path.join(ckpt_dir, "model")
            weights_path = os.path.join(ckpt_dir, "best.weights.h5")
            enc_path = os.path.join(ckpt_dir, "encoder.weights.h5")
            if os.path.isdir(model_path):
                candidates.append((model_path, meta_path))
            elif os.path.isfile(weights_path):
                candidates.append((weights_path, meta_path))
            elif os.path.isfile(enc_path):
                candidates.append((enc_path, meta_path))

    if not candidates:
        return None, None

    ckpt_path, meta_path = candidates[0]
    with open(meta_path) as f:
        meta = json.load(f)
    log.info(f"[{stage}] Found checkpoint: {ckpt_path} (cutoff={meta.get('cutoff_date')})")
    return ckpt_path, meta

def resolve_parent_checkpoint(stage, bucket_tag, symbol=None):
    """
    Given the current stage, find the most recent successful checkpoint
    from the parent stage in the hierarchy.
    Hierarchy: yearly -> monthly -> weekly -> daily
    Returns (checkpoint_path, meta_dict).
    If no parent exists, returns (None, None).
    """
    parent_stage = STAGE_PARENT.get(stage)
    if not parent_stage:
        return None, None   # yearly has no parent
    ckpt_path, meta = load_latest_successful_checkpoint(parent_stage, bucket_tag, symbol=symbol)
    if ckpt_path:
        log.info(f"[{stage}] Resolved parent checkpoint from '{parent_stage}': {ckpt_path}")
    else:
        # Try bucket-level (encoder) if symbol-level not found
        ckpt_path, meta = load_latest_successful_checkpoint(parent_stage, bucket_tag, symbol=None)
        if ckpt_path:
            log.info(f"[{stage}] Resolved bucket-level parent from '{parent_stage}': {ckpt_path}")
    return ckpt_path, meta

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def rsi(series, period=14):
    d    = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = -d.clip(upper=0).rolling(period).mean()
    rs   = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def clean_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                         .str.replace(',', '', regex=False)
                         .str.replace('%', '', regex=False))
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def nw_kernel_smooth(series, h=5, window=20):
    arr = series.values
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window)
        idx   = np.arange(start, i + 1)
        w     = np.exp(-0.5 * ((i - idx) / h) ** 2)
        out[i] = np.sum(w * arr[start:i+1]) / (np.sum(w) + 1e-9)
    return out

def cumulative_logret_forward(logret_series, horizon=1):
    s   = pd.Series(logret_series.reshape(-1))
    cum = s.rolling(window=horizon).sum().shift(-(horizon - 1))
    return cum.values.reshape(-1, 1)

def make_sequences_masked(X, y_dict, L, mask):
    idxs = [i for i in range(L, len(X)) if mask[i]]
    if len(idxs) == 0:
        out = {"X": np.zeros((0, L, X.shape[1]), dtype=X.dtype)}
        for k, arr in y_dict.items():
            out[k] = np.zeros((0, arr.shape[1]), dtype=arr.dtype)
        return out, []
    Xs  = np.stack([X[i-L:i] for i in idxs], axis=0)
    out = {"X": Xs}
    for k, arr in y_dict.items():
        out[k] = np.stack([arr[i] for i in idxs], axis=0)
    return out, idxs

def apply_label_smoothing(y, eps=LABEL_SMOOTH_EPS):
    return (1 - eps) * y + eps * 0.5

# ===============================================================
# ADVANCED FEATURE ECOSYSTEM HELPERS
# Raw indicators + statistics + calculus + ecosystem + probability
# ===============================================================

EPS = 1e-9

def _safe_div(a, b):
    return a / (b + EPS)

def _clip_series(s, lo=-1e6, hi=1e6):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan).clip(lo, hi)

def _rolling_z(s, window=20):
    s = pd.Series(s).astype(float)
    return (s - s.rolling(window).mean()) / (s.rolling(window).std() + EPS)

def _robust_z(s, window=20):
    s = pd.Series(s).astype(float)
    med = s.rolling(window).median()
    mad = (s - med).abs().rolling(window).median()
    return 0.6745 * (s - med) / (mad + EPS)

def _rolling_percentile(s, window=60):
    s = pd.Series(s).astype(float)
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan,
        raw=False
    )

def _rolling_entropy(s, window=20, bins=10):
    s = pd.Series(s).astype(float)

    def _ent(x):
        x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 5:
            return np.nan
        hist, _ = np.histogram(x, bins=bins)
        p = hist / (hist.sum() + EPS)
        p = p[p > 0]
        return -np.sum(p * np.log(p + EPS))

    return s.rolling(window).apply(_ent, raw=False)

def _rolling_autocorr(s, lag=1, window=20):
    s = pd.Series(s).astype(float)

    def _acf(x):
        x = pd.Series(x).dropna()
        if len(x) <= lag + 3:
            return np.nan
        return x.autocorr(lag=lag)

    return s.rolling(window).apply(_acf, raw=False)

def _hurst_approx(s, window=60):
    s = pd.Series(s).astype(float)

    def _hurst(x):
        x = pd.Series(x).dropna().values
        if len(x) < 30:
            return np.nan
        y = x - np.mean(x)
        z = np.cumsum(y)
        r = np.max(z) - np.min(z)
        sd = np.std(x) + EPS
        return np.log((r / sd) + EPS) / np.log(len(x) + EPS)

    return s.rolling(window).apply(_hurst, raw=False)

def _variance_ratio(s, window=20, lag=5):
    s = pd.Series(s).astype(float)

    def _vr(x):
        x = pd.Series(x).dropna()
        if len(x) < lag + 5:
            return np.nan
        var_1 = x.diff().var()
        var_q = x.diff(lag).var()
        return var_q / (lag * var_1 + EPS)

    return s.rolling(window + lag).apply(_vr, raw=False)

def _rolling_slope(s, window=20):
    s = pd.Series(s).astype(float)

    def _slope(x):
        x = pd.Series(x).dropna().values
        if len(x) < 5:
            return np.nan
        t = np.arange(len(x))
        coef = np.polyfit(t, x, 1)[0]
        return coef

    return s.rolling(window).apply(_slope, raw=False)

def _rolling_linear_r2(s, window=20):
    s = pd.Series(s).astype(float)

    def _r2(x):
        x = pd.Series(x).dropna().values
        if len(x) < 5:
            return np.nan
        t = np.arange(len(x))
        coef = np.polyfit(t, x, 1)
        pred = coef[0] * t + coef[1]
        ss_res = np.sum((x - pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2) + EPS
        return 1 - ss_res / ss_tot

    return s.rolling(window).apply(_r2, raw=False)

def _rolling_quadratic_curvature(s, window=20):
    s = pd.Series(s).astype(float)

    def _curv(x):
        x = pd.Series(x).dropna().values
        if len(x) < 8:
            return np.nan
        t = np.arange(len(x))
        coef = np.polyfit(t, x, 2)
        return 2 * coef[0]

    return s.rolling(window).apply(_curv, raw=False)

def _consecutive_condition_count(cond):
    cond = pd.Series(cond).fillna(False).astype(bool)
    out = np.zeros(len(cond), dtype=float)
    run = 0
    for i, v in enumerate(cond.values):
        run = run + 1 if v else 0
        out[i] = run
    return pd.Series(out, index=cond.index)

def _ema(s, span):
    return pd.Series(s).astype(float).ewm(span=span, adjust=False).mean()

def _wma(s, window):
    s = pd.Series(s).astype(float)
    weights = np.arange(1, window + 1)
    return s.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _hma(s, window=20):
    half = max(2, int(window / 2))
    sqrt_w = max(2, int(np.sqrt(window)))
    return _wma(2 * _wma(s, half) - _wma(s, window), sqrt_w)

def _kama(close, er_window=10, fast=2, slow=30):
    close = pd.Series(close).astype(float)
    change = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = change / (volatility + EPS)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.full(len(close), np.nan)
    if len(close) == 0:
        return pd.Series(kama, index=close.index)
    kama[0] = close.iloc[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1])
    return pd.Series(kama, index=close.index)

def _true_range(df):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def _adx_dmi(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.rolling(period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr * period + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr * period + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di

def _aroon(df, period=25):
    high = df["High"]
    low = df["Low"]

    aroon_up = high.rolling(period + 1).apply(
        lambda x: 100 * np.argmax(x) / period if len(x) > period else np.nan,
        raw=True
    )
    aroon_down = low.rolling(period + 1).apply(
        lambda x: 100 * np.argmin(x) / period if len(x) > period else np.nan,
        raw=True
    )
    return aroon_up, aroon_down, aroon_up - aroon_down

def _supertrend(df, period=10, multiplier=3.0):
    hl2 = (df["High"] + df["Low"]) / 2
    atr = _true_range(df).rolling(period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = pd.Series(index=df.index, dtype=float)
    supertrend = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lowerband.iloc[i]
            continue

        if upperband.iloc[i] < final_upper.iloc[i - 1] or df["Close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lowerband.iloc[i] > final_lower.iloc[i - 1] or df["Close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if df["Close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return supertrend, direction

def _choppiness_index(df, period=14):
    tr_sum = _true_range(df).rolling(period).sum()
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    return 100 * np.log10(tr_sum / (high_max - low_min + EPS)) / np.log10(period)

def _macd(close, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _tsi(close, long=25, short=13):
    mom = pd.Series(close).diff()
    abs_mom = mom.abs()
    tsi = 100 * _ema(_ema(mom, long), short) / (_ema(_ema(abs_mom, long), short) + EPS)
    return tsi

def _stochastic(df, period=14, smooth=3):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + EPS)
    d = k.rolling(smooth).mean()
    return k, d

def _ultimate_oscillator(df, p1=7, p2=14, p3=28):
    prev_close = df["Close"].shift(1)
    bp = df["Close"] - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    tr = pd.concat([df["High"], prev_close], axis=1).max(axis=1) - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)

    avg1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + EPS)
    avg2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + EPS)
    avg3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + EPS)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

def _connors_rsi(close, rsi_period=3, streak_rsi_period=2, rank_period=100):
    close = pd.Series(close).astype(float)
    rsi_close = rsi(close, rsi_period)

    up = close > close.shift(1)
    down = close < close.shift(1)

    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if up.iloc[i]:
            streak[i] = max(1, streak[i - 1] + 1)
        elif down.iloc[i]:
            streak[i] = min(-1, streak[i - 1] - 1)
        else:
            streak[i] = 0

    streak_rsi = rsi(pd.Series(streak, index=close.index), streak_rsi_period)
    roc1 = close.pct_change()
    percent_rank = roc1.rolling(rank_period).apply(
        lambda x: 100 * pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )
    return (rsi_close + streak_rsi + percent_rank) / 3

def _fisher_transform(df, period=10):
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()
    x = 2 * ((df["Close"] - low_min) / (high_max - low_min + EPS) - 0.5)
    x = x.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + x) / (1 - x + EPS))
    return fisher

def _obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()

def _cmf(df, period=20):
    mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + EPS)
    mf_vol = mf_mult * df["Volume"]
    return mf_vol.rolling(period).sum() / (df["Volume"].rolling(period).sum() + EPS)

def _mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = tp * df["Volume"]
    pos_flow = money_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0.0)
    mfr = pos_flow.rolling(period).sum() / (neg_flow.rolling(period).sum() + EPS)
    return 100 - (100 / (1 + mfr))

def _klinger(df, fast=34, slow=55, signal=13):
    hlc = df["High"] + df["Low"] + df["Close"]
    trend = np.where(hlc > hlc.shift(1), 1, -1)
    dm = df["High"] - df["Low"]
    cm = dm.copy()
    for i in range(1, len(df)):
        if trend[i] == trend[i - 1]:
            cm.iloc[i] = cm.iloc[i - 1] + dm.iloc[i]
        else:
            cm.iloc[i] = dm.iloc[i - 1] + dm.iloc[i]
    vf = df["Volume"] * trend * abs(2 * (dm / (cm + EPS) - 1)) * 100
    ko = _ema(vf, fast) - _ema(vf, slow)
    sig = _ema(ko, signal)
    return ko - sig

def _ease_of_movement(df, period=14):
    midpoint_move = ((df["High"] + df["Low"]) / 2).diff()
    box_ratio = df["Volume"] / (df["High"] - df["Low"] + EPS)
    eom = midpoint_move / (box_ratio + EPS)
    return eom.rolling(period).mean()

def _rolling_beta_alpha_corr(df, index_col="^NSEI", window=60):
    stock_ret = df["Close"].pct_change()
    if index_col in df.columns:
        market_ret = pd.to_numeric(df[index_col], errors="coerce").pct_change()
    else:
        market_ret = pd.Series(0.0, index=df.index)

    cov = stock_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    beta = cov / (var + EPS)
    alpha = stock_ret.rolling(window).mean() - beta * market_ret.rolling(window).mean()
    corr = stock_ret.rolling(window).corr(market_ret)
    return beta, alpha, corr

def _weekly_features(df):
    if "Date" not in df.columns:
        df["Weekly_RSI"] = 0.0
        df["Weekly_MACD_Hist"] = 0.0
        df["Weekly_EMA20_Slope"] = 0.0
        df["Weekly_Trend_State"] = 0.0
        df["Monthly_EMA20_Slope"] = 0.0
        return df

    temp = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    temp = temp.dropna(subset=["Date"]).set_index("Date")

    weekly = temp.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    if len(weekly) > 5:
        weekly["Weekly_RSI"] = rsi(weekly["Close"], 14)
        _, _, whist = _macd(weekly["Close"])
        weekly["Weekly_MACD_Hist"] = whist
        weekly["Weekly_EMA20"] = weekly["Close"].ewm(span=20, adjust=False).mean()
        weekly["Weekly_EMA20_Slope"] = weekly["Weekly_EMA20"].pct_change(3)
        weekly["Weekly_Trend_State"] = np.where(weekly["Weekly_EMA20_Slope"] > 0, 1.0, -1.0)

        weekly_map = weekly[["Weekly_RSI", "Weekly_MACD_Hist", "Weekly_EMA20_Slope", "Weekly_Trend_State"]]
        df = pd.merge_asof(
            df.sort_values("Date"),
            weekly_map.reset_index().sort_values("Date"),
            on="Date",
            direction="backward"
        )
    else:
        df["Weekly_RSI"] = 0.0
        df["Weekly_MACD_Hist"] = 0.0
        df["Weekly_EMA20_Slope"] = 0.0
        df["Weekly_Trend_State"] = 0.0

    monthly = temp.resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    if len(monthly) > 5:
        monthly["Monthly_EMA20"] = monthly["Close"].ewm(span=20, adjust=False).mean()
        monthly["Monthly_EMA20_Slope"] = monthly["Monthly_EMA20"].pct_change(3)
        month_map = monthly[["Monthly_EMA20_Slope"]]
        df = pd.merge_asof(
            df.sort_values("Date"),
            month_map.reset_index().rename(columns={"Date": "Date"}).sort_values("Date"),
            on="Date",
            direction="backward"
        )
    else:
        df["Monthly_EMA20_Slope"] = 0.0

    return df

def add_raw_advanced_features(df, sec_idx_col=None):
    df = df.copy()

    # Trend / regime
    df["ADX14"], df["PlusDI14"], df["MinusDI14"] = _adx_dmi(df, 14)
    df["DI_Spread"] = df["PlusDI14"] - df["MinusDI14"]

    df["Aroon_Up"], df["Aroon_Down"], df["Aroon_Osc"] = _aroon(df, 25)
    df["Choppiness_Index"] = _choppiness_index(df, 14)

    df["Supertrend"], df["Supertrend_Direction"] = _supertrend(df, 10, 3.0)
    df["HMA20"] = _hma(df["Close"], 20)
    df["KAMA20"] = _kama(df["Close"], 10, 2, 30)

    ema1 = _ema(df["Close"], 15)
    ema2 = _ema(ema1, 15)
    ema3 = _ema(ema2, 15)
    df["TRIX"] = ema3.pct_change() * 100

    # Momentum
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(df["Close"])
    ppo = (_ema(df["Close"], 12) - _ema(df["Close"], 26)) / (_ema(df["Close"], 26) + EPS) * 100
    df["PPO"] = ppo
    df["ROC10"] = df["Close"].pct_change(10) * 100
    df["ROC20"] = df["Close"].pct_change(20) * 100
    df["TSI"] = _tsi(df["Close"])
    df["Stoch_K"], df["Stoch_D"] = _stochastic(df, 14, 3)
    df["WilliamsR"] = -100 * (df["High"].rolling(14).max() - df["Close"]) / (
        df["High"].rolling(14).max() - df["Low"].rolling(14).min() + EPS
    )
    df["Ultimate_Oscillator"] = _ultimate_oscillator(df)
    df["Connors_RSI"] = _connors_rsi(df["Close"])
    df["Fisher_Transform"] = _fisher_transform(df)

    # Volume / money flow
    df["OBV"] = _obv(df)
    df["OBV_Slope"] = _rolling_slope(df["OBV"], 10)
    df["CMF20"] = _cmf(df, 20)
    df["MFI14"] = _mfi(df, 14)
    df["PVT"] = (df["Volume"] * df["Close"].pct_change()).fillna(0).cumsum()
    df["Klinger"] = _klinger(df)
    df["Ease_Of_Movement"] = _ease_of_movement(df)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / (df["Volume"].cumsum() + EPS)
    df["VWAP_Distance"] = (df["Close"] - df["VWAP"]) / (df["VWAP"] + EPS)

    df["Volume_Delta"] = np.where(df["Close"] >= df["Close"].shift(1), df["Volume"], -df["Volume"])
    up_vol = df["Volume"].where(df["Close"] >= df["Close"].shift(1), 0.0).rolling(20).sum()
    down_vol = df["Volume"].where(df["Close"] < df["Close"].shift(1), 0.0).rolling(20).sum()
    df["UpDown_Volume_Ratio"] = up_vol / (down_vol + EPS)

    # Volatility / compression
    df["ATR_Pct"] = df["ATR"] / (df["Close"] + EPS)
    df["Historical_Volatility_20"] = df["LogRet"].rolling(20).std() * np.sqrt(252)
    df["Historical_Volatility_60"] = df["LogRet"].rolling(60).std() * np.sqrt(252)

    df["BB_Width"] = (df["BB_Up"] - df["BB_Lo"]) / (df["BB_Mid"] + EPS)
    df["BB_Squeeze"] = (df["BB_Width"] < df["BB_Width"].rolling(60).quantile(0.20)).astype(float)

    df["Donchian_High20"] = df["High"].rolling(20).max()
    df["Donchian_Low20"] = df["Low"].rolling(20).min()
    df["Donchian_Width"] = (df["Donchian_High20"] - df["Donchian_Low20"]) / (df["Close"] + EPS)
    df["Donchian_Pos"] = (df["Close"] - df["Donchian_Low20"]) / (
        df["Donchian_High20"] - df["Donchian_Low20"] + EPS
    )

    df["Parkinson_Volatility"] = np.sqrt(
        (1.0 / (4.0 * np.log(2))) *
        (np.log(df["High"] / (df["Low"] + EPS)) ** 2).rolling(20).mean()
    )

    log_hl = np.log(df["High"] / (df["Low"] + EPS))
    log_co = np.log(df["Close"] / (df["Open"] + EPS))
    df["Garman_Klass_Volatility"] = np.sqrt(
        (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(20).mean().clip(lower=0)
    )

    roll_max = df["Close"].rolling(20).max()
    drawdown = (df["Close"] - roll_max) / (roll_max + EPS)
    df["Ulcer_Index"] = np.sqrt((drawdown.clip(upper=0) ** 2).rolling(20).mean())

    # Market structure
    df["Distance_From_20D_High"] = (df["Close"] - df["Donchian_High20"]) / (df["Donchian_High20"] + EPS)
    df["Distance_From_20D_Low"] = (df["Close"] - df["Donchian_Low20"]) / (df["Donchian_Low20"] + EPS)

    high_52 = df["High"].rolling(252).max()
    low_52 = df["Low"].rolling(252).min()
    df["Distance_From_52W_High"] = (df["Close"] - high_52) / (high_52 + EPS)
    df["Distance_From_52W_Low"] = (df["Close"] - low_52) / (low_52 + EPS)

    df["Breakout_Distance"] = (df["Close"] - df["Donchian_High20"].shift(1)) / (
        df["ATR"].rolling(20).mean() + EPS
    )
    df["Pullback_Depth"] = (df["High"].rolling(20).max() - df["Close"]) / (
        df["ATR"].rolling(20).mean() + EPS
    )

    # Relative strength
    if "^NSEI" in df.columns:
        nifty_ret = pd.to_numeric(df["^NSEI"], errors="coerce").pct_change()
        nifty_5 = pd.to_numeric(df["^NSEI"], errors="coerce").pct_change(5)
        nifty_20 = pd.to_numeric(df["^NSEI"], errors="coerce").pct_change(20)
    else:
        nifty_ret = pd.Series(0.0, index=df.index)
        nifty_5 = pd.Series(0.0, index=df.index)
        nifty_20 = pd.Series(0.0, index=df.index)

    stock_5 = df["Close"].pct_change(5)
    stock_20 = df["Close"].pct_change(20)
    df["RS_Nifty_5D"] = stock_5 - nifty_5
    df["RS_Nifty_20D"] = stock_20 - nifty_20

    if sec_idx_col and sec_idx_col in df.columns:
        sec_series = pd.to_numeric(df[sec_idx_col], errors="coerce")
        df["RS_Sector_5D"] = stock_5 - sec_series.pct_change(5)
        df["RS_Sector_20D"] = stock_20 - sec_series.pct_change(20)
    else:
        df["RS_Sector_5D"] = 0.0
        df["RS_Sector_20D"] = 0.0

    df["Rolling_Beta_60"], df["Rolling_Alpha_20"], df["Rolling_Correlation_Nifty_60"] = _rolling_beta_alpha_corr(
        df, "^NSEI", 60
    )
    df["Rolling_Alpha_20"] = df["Close"].pct_change().rolling(20).mean() - nifty_ret.rolling(20).mean()
    tracking_err = (df["Close"].pct_change() - nifty_ret).rolling(60).std()
    df["Information_Ratio_60"] = (df["Close"].pct_change() - nifty_ret).rolling(60).mean() / (tracking_err + EPS)

    df["RS_Persistence_5"] = _consecutive_condition_count(df["RS_Nifty_5D"] > 0)
    df["RS_Persistence_20"] = _consecutive_condition_count(df["RS_Nifty_20D"] > 0)
    df["Alpha_Persistence_20"] = _consecutive_condition_count(df["Rolling_Alpha_20"] > 0)

    # Cross-timeframe
    df = _weekly_features(df)
    df["Daily_Inside_Weekly_Trend"] = np.sign(df["Ret5"].fillna(0)) * np.sign(df["Weekly_EMA20_Slope"].fillna(0))
    df["Daily_Weekly_Momentum_Alignment"] = np.sign(df["MACD_Hist"].fillna(0)) * np.sign(df["Weekly_MACD_Hist"].fillna(0))

    # Liquidity / tradability
    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    df["Amihud_Illiquidity"] = df["LogRet"].abs() / (df["Dollar_Volume"] + EPS)
    df["Volume_Dryup_Ratio"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(20).mean() + EPS)
    df["Liquidity_Shock"] = _rolling_z(df["Dollar_Volume"], 20)
    df["Spread_Proxy"] = (df["High"] - df["Low"]) / (df["Close"] + EPS)

    # Gap behavior
    df["Gap_vs_ATR"] = (df["Open"] - df["Close"].shift(1)) / (df["ATR"].rolling(20).mean() + EPS)
    gap_up = df["Open"] > df["Close"].shift(1)
    gap_down = df["Open"] < df["Close"].shift(1)
    gap_filled_up = gap_up & (df["Low"] <= df["Close"].shift(1))
    gap_filled_down = gap_down & (df["High"] >= df["Close"].shift(1))
    df["Gap_Fill_Ratio"] = (gap_filled_up | gap_filled_down).astype(float).rolling(20).mean()
    df["Gap_Continuation_Flag"] = np.where(
        gap_up & (df["Close"] > df["Open"]), 1.0,
        np.where(gap_down & (df["Close"] < df["Open"]), -1.0, 0.0)
    )
    df["Gap_Exhaustion_Score"] = df["Gap_vs_ATR"].abs() * (1 - df["CandlePos"].clip(0, 1))
    df["Opening_Gap_Strength"] = df["Gap_vs_ATR"] * df["VolRel"]

    # Tail-risk / drawdown
    df["Downside_Semivariance_20"] = (df["LogRet"].clip(upper=0) ** 2).rolling(20).mean()
    roll_max20 = df["Close"].rolling(20).max()
    roll_max60 = df["Close"].rolling(60).max()
    df["Max_Drawdown_20"] = (df["Close"] - roll_max20) / (roll_max20 + EPS)
    df["Max_Drawdown_60"] = (df["Close"] - roll_max60) / (roll_max60 + EPS)
    df["Drawdown_Speed"] = df["Max_Drawdown_20"].diff(5)
    df["Left_Tail_Return_Count_20"] = (df["LogRet"] < -2 * df["Vol20"]).astype(float).rolling(20).sum()
    df["Crash_Risk_Score"] = (
        df["Left_Tail_Return_Count_20"] / 20.0
        + df["Downside_Semivariance_20"].rank(pct=True)
        + (-df["Max_Drawdown_20"]).clip(lower=0)
    )

    # Exhaustion
    df["Trend_Age"] = _consecutive_condition_count(df["Close"] > df["EMA20"])
    df["Consecutive_Up_Days"] = _consecutive_condition_count(df["Close"] > df["Close"].shift(1))
    df["Consecutive_Down_Days"] = _consecutive_condition_count(df["Close"] < df["Close"].shift(1))
    df["Distance_From_EMA20_ATR"] = (df["Close"] - df["EMA20"]) / (df["ATR"].rolling(20).mean() + EPS)
    df["Distance_From_EMA50_ATR"] = (df["Close"] - df["EMA50"]) / (df["ATR"].rolling(20).mean() + EPS)
    df["Overextension_Score"] = (
        df["Distance_From_EMA20_ATR"].abs()
        + df["RSI14"].sub(50).abs() / 50.0
        + df["Donchian_Pos"].sub(0.5).abs()
    )

    # Compression / expansion
    df["Range_Compression_5_20"] = df["Range"].rolling(5).mean() / (df["Range"].rolling(20).mean() + EPS)
    df["Volume_Compression_5_20"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(20).mean() + EPS)
    df["Volatility_Compression_20"] = df["Vol20"] / (df["Vol20"].rolling(60).mean() + EPS)
    df["Squeeze_Intensity"] = (
        (1 - df["BB_Width"].rank(pct=True)).clip(0, 1)
        + (1 - df["Range_Compression_5_20"]).clip(0, 1)
    )
    df["Expansion_Breakout_Score"] = df["Squeeze_Intensity"] * df["BreakReliab"] * np.maximum(df["Donchian_Pos"], 0)

    # Candle sequence
    df["Bullish_Candle_Streak"] = _consecutive_condition_count(df["Close"] > df["Open"])
    df["Bearish_Candle_Streak"] = _consecutive_condition_count(df["Close"] < df["Open"])
    df["Higher_Close_Count_5"] = (df["Close"] > df["Close"].shift(1)).astype(float).rolling(5).sum()
    df["Lower_Close_Count_5"] = (df["Close"] < df["Close"].shift(1)).astype(float).rolling(5).sum()
    inside_bar = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    df["Inside_Bar_Count_10"] = inside_bar.astype(float).rolling(10).sum()
    df["Wide_Range_Bar_Flag"] = (df["Range"] > df["Range"].rolling(20).quantile(0.80)).astype(float)
    df["Narrow_Range_Bar_Flag"] = (df["Range"] < df["Range"].rolling(20).quantile(0.20)).astype(float)

    return df

def add_raw_statistics_and_calculus(df):
    df = df.copy()

    # Statistics
    df["Robust_Return_Z20"] = _robust_z(df["LogRet"], 20)
    df["Rolling_Median_Return_20"] = df["LogRet"].rolling(20).median()
    df["Rolling_MAD_Return_20"] = (df["LogRet"] - df["Rolling_Median_Return_20"]).abs().rolling(20).median()
    df["Return_IQR_60"] = df["LogRet"].rolling(60).quantile(0.75) - df["LogRet"].rolling(60).quantile(0.25)
    df["Rolling_Skew_20"] = df["LogRet"].rolling(20).skew()
    df["Rolling_Kurtosis_20"] = df["LogRet"].rolling(20).kurt()
    df["Rolling_Sharpe_20"] = df["LogRet"].rolling(20).mean() / (df["LogRet"].rolling(20).std() + EPS)
    df["Rolling_TStat_Return_20"] = df["LogRet"].rolling(20).mean() / (
        df["LogRet"].rolling(20).std() / np.sqrt(20) + EPS
    )

    df["Entropy_Return_20"] = _rolling_entropy(df["LogRet"], 20)
    df["Autocorr_Return_1_20"] = _rolling_autocorr(df["LogRet"], 1, 20)
    df["Autocorr_Return_5_60"] = _rolling_autocorr(df["LogRet"], 5, 60)
    df["Hurst_Exponent_60"] = _hurst_approx(df["LogRet"], 60)
    df["Variance_Ratio_20"] = _variance_ratio(df["LogRet"], 20, 5)

    df["RSI_Percentile_60"] = _rolling_percentile(df["RSI14"], 60)
    df["ATR_Percentile_60"] = _rolling_percentile(df["ATR_Pct"], 60)
    df["Volume_Percentile_60"] = _rolling_percentile(df["Volume"], 60)
    df["Range_Percentile_60"] = _rolling_percentile(df["Range"], 60)
    df["Volatility_Percentile_60"] = _rolling_percentile(df["Vol20"], 60)

    if "^NSEI" in df.columns:
        nifty_ret = pd.to_numeric(df["^NSEI"], errors="coerce").pct_change()
        df["Rolling_Cov_Stock_Nifty_60"] = df["LogRet"].rolling(60).cov(nifty_ret)
    else:
        df["Rolling_Cov_Stock_Nifty_60"] = 0.0

    df["Beta_Stability_60"] = df["Rolling_Beta_60"].rolling(60).std()

    # Calculus / dynamics
    df["Price_Slope_5"] = _rolling_slope(df["Close"], 5)
    df["Price_Slope_20"] = _rolling_slope(df["Close"], 20)
    df["EMA20_Slope"] = df["EMA20"].pct_change(5)
    df["EMA50_Slope"] = df["EMA50"].pct_change(5)
    df["EMA20_Acceleration"] = df["EMA20_Slope"].diff(5)

    df["Rolling_Linear_Trend_R2_20"] = _rolling_linear_r2(df["Close"], 20)
    df["Rolling_Quadratic_Curvature_20"] = _rolling_quadratic_curvature(df["Close"], 20)
    df["Trend_Convexity_20"] = df["Rolling_Quadratic_Curvature_20"]
    df["Price_Inflection_Flag"] = (
        np.sign(df["Rolling_Quadratic_Curvature_20"]) != np.sign(df["Rolling_Quadratic_Curvature_20"].shift(1))
    ).astype(float)

    df["RSI_Velocity"] = df["RSI14"].diff()
    df["RSI_Acceleration"] = df["RSI_Velocity"].diff()
    df["MACD_Hist_Velocity"] = df["MACD_Hist"].diff()
    df["MACD_Hist_Acceleration"] = df["MACD_Hist_Velocity"].diff()

    df["RSI_Turning_Point"] = (np.sign(df["RSI_Velocity"]) != np.sign(df["RSI_Velocity"].shift(1))).astype(float)
    df["MACD_Hist_Turning_Point"] = (
        np.sign(df["MACD_Hist_Velocity"]) != np.sign(df["MACD_Hist_Velocity"].shift(1))
    ).astype(float)

    df["OBV_Velocity"] = df["OBV"].diff()
    df["CMF_Velocity"] = df["CMF20"].diff()
    df["ATR_Velocity"] = df["ATR_Pct"].diff()
    df["Volatility_Slope_20"] = _rolling_slope(df["Vol20"], 20)
    df["Volatility_Acceleration_20"] = df["Volatility_Slope_20"].diff()
    df["Volume_Acceleration"] = df["Volume"].diff().diff()

    df["Drawdown_Velocity"] = df["Max_Drawdown_20"].diff()
    df["Drawdown_Acceleration"] = df["Drawdown_Velocity"].diff()

    return df

def _squash(s):
    s = pd.Series(s).astype(float)
    return np.tanh(_rolling_z(s, 60).fillna(0))

def add_family_ecosystem_features(df):
    df = df.copy()

    # Family scores in approximately [-1, +1]
    df["Trend_Score"] = np.nanmean(np.vstack([
        _squash(df["EMA20_Slope"]),
        _squash(df["DI_Spread"]),
        _squash(df["Aroon_Osc"]),
        df["Supertrend_Direction"].fillna(0),
        _squash(df["MACD_Hist"])
    ]), axis=0)

    df["Momentum_Score"] = np.nanmean(np.vstack([
        _squash(df["RSI14"] - 50),
        _squash(df["MACD_Hist"]),
        _squash(df["ROC10"]),
        _squash(df["TSI"]),
        _squash(df["Fisher_Transform"])
    ]), axis=0)

    df["VolumeFlow_Score"] = np.nanmean(np.vstack([
        _squash(df["OBV_Slope"]),
        _squash(df["CMF20"]),
        _squash(df["MFI14"] - 50),
        _squash(df["PVT"]),
        _squash(df["Volume_Delta"])
    ]), axis=0)

    df["Volatility_Score"] = np.nanmean(np.vstack([
        _squash(df["ATR_Pct"]),
        _squash(df["Historical_Volatility_20"]),
        _squash(df["BB_Width"]),
        -_squash(df["Choppiness_Index"])
    ]), axis=0)

    df["Structure_Score"] = np.nanmean(np.vstack([
        _squash(df["Donchian_Pos"] - 0.5),
        _squash(df["BB_Percent"] - 0.5),
        _squash(df["KC_Pos"] - 0.5),
        _squash(df["BreakReliab"])
    ]), axis=0)

    df["RelativeStrength_Score"] = np.nanmean(np.vstack([
        _squash(df["RS_Nifty_20D"]),
        _squash(df["RS_Sector_20D"]),
        _squash(df["Rolling_Alpha_20"]),
        _squash(df["Information_Ratio_60"])
    ]), axis=0)

    df["CrossTimeframe_Score"] = np.nanmean(np.vstack([
        _squash(df["Weekly_RSI"] - 50),
        _squash(df["Weekly_MACD_Hist"]),
        _squash(df["Weekly_EMA20_Slope"]),
        df["Daily_Weekly_Momentum_Alignment"].fillna(0)
    ]), axis=0)

    df["Liquidity_Score"] = np.nanmean(np.vstack([
        _squash(df["Dollar_Volume"]),
        -_squash(df["Amihud_Illiquidity"]),
        _squash(df["Volume_Dryup_Ratio"]),
        -_squash(df["Spread_Proxy"])
    ]), axis=0)

    df["Gap_Score"] = np.nanmean(np.vstack([
        _squash(df["Gap_vs_ATR"]),
        _squash(df["Gap_Continuation_Flag"]),
        -_squash(df["Gap_Exhaustion_Score"])
    ]), axis=0)

    df["RiskDrawdown_Score"] = -np.nanmean(np.vstack([
        _squash(df["Crash_Risk_Score"]),
        _squash(-df["Max_Drawdown_20"]),
        _squash(df["Left_Tail_Return_Count_20"]),
        _squash(df["Downside_Semivariance_20"])
    ]), axis=0)

    df["Exhaustion_Score"] = -np.nanmean(np.vstack([
        _squash(df["Overextension_Score"]),
        _squash(df["Consecutive_Up_Days"]),
        _squash(df["Distance_From_EMA20_ATR"].abs())
    ]), axis=0)

    df["Compression_Score"] = np.nanmean(np.vstack([
        -_squash(df["Range_Compression_5_20"]),
        -_squash(df["Volume_Compression_5_20"]),
        -_squash(df["Volatility_Compression_20"]),
        _squash(df["Squeeze_Intensity"])
    ]), axis=0)

    df["CandleSequence_Score"] = np.nanmean(np.vstack([
        _squash(df["Bullish_Candle_Streak"] - df["Bearish_Candle_Streak"]),
        _squash(df["Higher_Close_Count_5"] - df["Lower_Close_Count_5"]),
        _squash(df["Wide_Range_Bar_Flag"] - df["Narrow_Range_Bar_Flag"])
    ]), axis=0)

    families = [
        "Trend", "Momentum", "VolumeFlow", "Volatility", "Structure",
        "RelativeStrength", "CrossTimeframe", "Liquidity", "Gap",
        "RiskDrawdown", "Exhaustion", "Compression", "CandleSequence"
    ]

    for fam in families:
        col = f"{fam}_Score"

        # Family statistics
        df[f"{fam}_Z20"] = _rolling_z(df[col], 20)
        df[f"{fam}_Z60"] = _rolling_z(df[col], 60)
        df[f"{fam}_Percentile_60"] = _rolling_percentile(df[col], 60)
        df[f"{fam}_RollingMean_20"] = df[col].rolling(20).mean()
        df[f"{fam}_RollingStd_20"] = df[col].rolling(20).std()
        df[f"{fam}_Autocorr_20"] = _rolling_autocorr(df[col], 1, 20)
        df[f"{fam}_Persistence_20"] = _consecutive_condition_count(df[col] > 0)
        df[f"{fam}_Signal_Stability"] = 1 / (df[col].rolling(20).std() + EPS)
        df[f"{fam}_Noise_Ratio"] = df[col].diff().abs().rolling(20).mean() / (
            df[col].abs().rolling(20).mean() + EPS
        )
        df[f"{fam}_Price_Correlation_20"] = df[col].rolling(20).corr(df["LogRet"])
        df[f"{fam}_Price_Correlation_60"] = df[col].rolling(60).corr(df["LogRet"])
        df[f"{fam}_Lead_Return_Correlation_20"] = df[col].rolling(20).corr(df["LogRet"].shift(-1))

        # Family calculus / dynamics
        df[f"{fam}_Velocity"] = df[col].diff()
        df[f"{fam}_Acceleration"] = df[f"{fam}_Velocity"].diff()
        df[f"{fam}_Curvature"] = _rolling_quadratic_curvature(df[col], 20)
        df[f"{fam}_Turning_Point"] = (
            np.sign(df[f"{fam}_Velocity"]) != np.sign(df[f"{fam}_Velocity"].shift(1))
        ).astype(float)
        df[f"{fam}_Slope_5"] = _rolling_slope(df[col], 5)
        df[f"{fam}_Slope_20"] = _rolling_slope(df[col], 20)
        df[f"{fam}_Divergence_From_Price"] = df[col] - _squash(df["LogRet"].rolling(5).sum())

    # Family-to-family relations
    df["Trend_Momentum_Agreement"] = df["Trend_Score"] * df["Momentum_Score"]
    df["Trend_Volume_Confirmation"] = df["Trend_Score"] * df["VolumeFlow_Score"]
    df["Trend_Volatility_Compatibility"] = df["Trend_Score"] * (1 - df["Volatility_Score"].abs())
    df["Trend_Structure_Alignment"] = df["Trend_Score"] * df["Structure_Score"]
    df["Trend_RelativeStrength_Alignment"] = df["Trend_Score"] * df["RelativeStrength_Score"]
    df["Trend_CrossTimeframe_Alignment"] = df["Trend_Score"] * df["CrossTimeframe_Score"]

    df["Momentum_Volume_Confirmation"] = df["Momentum_Score"] * df["VolumeFlow_Score"]
    df["Momentum_Volatility_Compatibility"] = df["Momentum_Score"] * (1 - df["Volatility_Score"].abs())
    df["Momentum_Structure_Alignment"] = df["Momentum_Score"] * df["Structure_Score"]
    df["Momentum_RelativeStrength_Alignment"] = df["Momentum_Score"] * df["RelativeStrength_Score"]
    df["Momentum_CrossTimeframe_Alignment"] = df["Momentum_Score"] * df["CrossTimeframe_Score"]

    df["Volume_Volatility_Pressure"] = df["VolumeFlow_Score"] * df["Volatility_Score"]
    df["Volume_Structure_Confirmation"] = df["VolumeFlow_Score"] * df["Structure_Score"]
    df["Volume_RelativeStrength_Confirmation"] = df["VolumeFlow_Score"] * df["RelativeStrength_Score"]
    df["Volume_Liquidity_Quality"] = df["VolumeFlow_Score"] * df["Liquidity_Score"]

    df["Volatility_Structure_Pressure"] = df["Volatility_Score"] * df["Structure_Score"]
    df["Volatility_Compression_Pressure"] = df["Volatility_Score"] * df["Compression_Score"]
    df["Volatility_Risk_Alignment"] = df["Volatility_Score"] * df["RiskDrawdown_Score"]
    df["Volatility_Exhaustion_Risk"] = df["Volatility_Score"] * df["Exhaustion_Score"]

    df["Structure_RelativeStrength_Alignment"] = df["Structure_Score"] * df["RelativeStrength_Score"]
    df["Structure_CrossTimeframe_Alignment"] = df["Structure_Score"] * df["CrossTimeframe_Score"]
    df["Structure_Compression_Breakout_Readiness"] = df["Structure_Score"] * df["Compression_Score"]
    df["Structure_Gap_Compatibility"] = df["Structure_Score"] * df["Gap_Score"]

    df["Risk_Trend_Conflict"] = -df["RiskDrawdown_Score"] * df["Trend_Score"]
    df["Risk_Momentum_Conflict"] = -df["RiskDrawdown_Score"] * df["Momentum_Score"]
    df["Exhaustion_Trend_Conflict"] = -df["Exhaustion_Score"] * df["Trend_Score"]
    df["Exhaustion_Momentum_Conflict"] = -df["Exhaustion_Score"] * df["Momentum_Score"]

    positive_family_cols = [
        "Trend_Score", "Momentum_Score", "VolumeFlow_Score", "Structure_Score",
        "RelativeStrength_Score", "CrossTimeframe_Score", "Liquidity_Score"
    ]

    df["Ecosystem_Agreement_Index"] = df[positive_family_cols].mean(axis=1)
    df["Ecosystem_Conflict_Index"] = df[positive_family_cols].std(axis=1)
    df["Ecosystem_Directional_Bias"] = np.tanh(df["Ecosystem_Agreement_Index"] - 0.5 * df["Ecosystem_Conflict_Index"])
    df["Ecosystem_Breakout_Readiness"] = (
        df["Compression_Score"].clip(lower=0)
        * df["Structure_Score"].clip(lower=0)
        * df["VolumeFlow_Score"].clip(lower=0)
    )
    df["Ecosystem_Reversal_Risk"] = (
        (-df["Exhaustion_Score"]).clip(lower=0)
        * (-df["RiskDrawdown_Score"]).clip(lower=0)
        * df["Volatility_Score"].abs()
    )
    df["Ecosystem_Noise_Level"] = df[
        ["Trend_Noise_Ratio", "Momentum_Noise_Ratio", "VolumeFlow_Noise_Ratio", "Volatility_Noise_Ratio"]
    ].mean(axis=1)

    return df

def add_indicator_internal_dots(df):
    df = df.copy()

    # =========================
    # RSI / Momentum dots
    # =========================
    df["RSI_Above_50_State"] = (df["RSI14"] > 50).astype(float)
    df["RSI_50_Cross_Up"] = ((df["RSI14"] > 50) & (df["RSI14"].shift(1) <= 50)).astype(float)
    df["RSI_50_Cross_Down"] = ((df["RSI14"] < 50) & (df["RSI14"].shift(1) >= 50)).astype(float)

    df["RSI_Above_55_State"] = (df["RSI14"] > 55).astype(float)
    df["RSI_Below_45_State"] = (df["RSI14"] < 45).astype(float)
    df["RSI_Neutral_45_55"] = ((df["RSI14"] >= 45) & (df["RSI14"] <= 55)).astype(float)
    df["RSI_Escape_Above_55"] = ((df["RSI14"] > 55) & (df["RSI14"].shift(1) <= 55)).astype(float)
    df["RSI_Escape_Below_45"] = ((df["RSI14"] < 45) & (df["RSI14"].shift(1) >= 45)).astype(float)

    df["RSI_MA14"] = df["RSI14"].rolling(14).mean()
    df["RSI_MA_Spread"] = df["RSI14"] - df["RSI_MA14"]
    df["RSI_MA_Cross_Up"] = ((df["RSI14"] > df["RSI_MA14"]) & (df["RSI14"].shift(1) <= df["RSI_MA14"].shift(1))).astype(float)
    df["RSI_MA_Cross_Down"] = ((df["RSI14"] < df["RSI_MA14"]) & (df["RSI14"].shift(1) >= df["RSI_MA14"].shift(1))).astype(float)

    df["RSI_Overbought_70"] = (df["RSI14"] > 70).astype(float)
    df["RSI_Oversold_30"] = (df["RSI14"] < 30).astype(float)
    df["RSI_Exit_Overbought"] = ((df["RSI14"] < 70) & (df["RSI14"].shift(1) >= 70)).astype(float)
    df["RSI_Exit_Oversold"] = ((df["RSI14"] > 30) & (df["RSI14"].shift(1) <= 30)).astype(float)

    if "RSI_Velocity" not in df.columns:
        df["RSI_Velocity"] = df["RSI14"].diff()
    if "RSI_Acceleration" not in df.columns:
        df["RSI_Acceleration"] = df["RSI_Velocity"].diff()

    df["RSI_Velocity_Cross_Up"] = ((df["RSI_Velocity"] > 0) & (df["RSI_Velocity"].shift(1) <= 0)).astype(float)
    df["RSI_Velocity_Cross_Down"] = ((df["RSI_Velocity"] < 0) & (df["RSI_Velocity"].shift(1) >= 0)).astype(float)
    df["RSI_Acceleration_Positive"] = (df["RSI_Acceleration"] > 0).astype(float)
    df["RSI_Acceleration_Negative"] = (df["RSI_Acceleration"] < 0).astype(float)

    df["RSI_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["RSI_Above_55_State"],
        df["RSI_Escape_Above_55"],
        df["RSI_MA_Cross_Up"],
        df["RSI_Velocity_Cross_Up"],
        df["RSI_Acceleration_Positive"]
    ]), axis=0)

    df["RSI_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["RSI_Below_45_State"],
        df["RSI_Escape_Below_45"],
        df["RSI_MA_Cross_Down"],
        df["RSI_Velocity_Cross_Down"],
        df["RSI_Acceleration_Negative"]
    ]), axis=0)

    df["RSI_Exhaustion_Evidence"] = np.nanmean(np.vstack([
        df["RSI_Overbought_70"],
        df["RSI_Exit_Overbought"],
        df["RSI14"].sub(70).clip(lower=0) / 30.0
    ]), axis=0)

    df["RSI_Reversal_Evidence"] = np.nanmean(np.vstack([
        df["RSI_Exit_Oversold"],
        df["RSI_Oversold_30"],
        (30 - df["RSI14"]).clip(lower=0) / 30.0
    ]), axis=0)

    # =========================
    # EMA / Trend dots
    # =========================
    atr_base = df["ATR"].rolling(20).mean() if "ATR" in df.columns else df["Close"].rolling(20).std()

    df["Price_Above_EMA20"] = (df["Close"] > df["EMA20"]).astype(float)
    df["Price_Above_EMA50"] = (df["Close"] > df["EMA50"]).astype(float)
    df["EMA20_Above_EMA50"] = (df["EMA20"] > df["EMA50"]).astype(float)
    df["EMA20_EMA50_Cross_Up"] = ((df["EMA20"] > df["EMA50"]) & (df["EMA20"].shift(1) <= df["EMA50"].shift(1))).astype(float)
    df["EMA20_EMA50_Cross_Down"] = ((df["EMA20"] < df["EMA50"]) & (df["EMA20"].shift(1) >= df["EMA50"].shift(1))).astype(float)

    if "EMA20_Slope" not in df.columns:
        df["EMA20_Slope"] = df["EMA20"].pct_change(5)
    if "EMA50_Slope" not in df.columns:
        df["EMA50_Slope"] = df["EMA50"].pct_change(5)
    if "EMA20_Acceleration" not in df.columns:
        df["EMA20_Acceleration"] = df["EMA20_Slope"].diff(5)

    df["EMA20_Slope_Positive"] = (df["EMA20_Slope"] > 0).astype(float)
    df["EMA50_Slope_Positive"] = (df["EMA50_Slope"] > 0).astype(float)
    df["EMA20_Acceleration_Positive"] = (df["EMA20_Acceleration"] > 0).astype(float)
    df["EMA20_Acceleration_Negative"] = (df["EMA20_Acceleration"] < 0).astype(float)

    df["EMA20_Distance_ATR"] = (df["Close"] - df["EMA20"]) / (atr_base + EPS)
    df["EMA50_Distance_ATR"] = (df["Close"] - df["EMA50"]) / (atr_base + EPS)
    df["EMA20_Overextended_Up"] = (df["EMA20_Distance_ATR"] > 2.0).astype(float)
    df["EMA20_Overextended_Down"] = (df["EMA20_Distance_ATR"] < -2.0).astype(float)

    df["Trend_Age_Bullish"] = _consecutive_condition_count(df["Close"] > df["EMA20"])
    df["Trend_Age_Bearish"] = _consecutive_condition_count(df["Close"] < df["EMA20"])
    df["Fresh_Bullish_Trend"] = ((df["Trend_Age_Bullish"] >= 1) & (df["Trend_Age_Bullish"] <= 5)).astype(float)
    df["Fresh_Bearish_Trend"] = ((df["Trend_Age_Bearish"] >= 1) & (df["Trend_Age_Bearish"] <= 5)).astype(float)
    df["Mature_Bullish_Trend"] = (df["Trend_Age_Bullish"] > 20).astype(float)
    df["Mature_Bearish_Trend"] = (df["Trend_Age_Bearish"] > 20).astype(float)

    df["EMA_Bullish_Trend_Evidence"] = np.nanmean(np.vstack([
        df["Price_Above_EMA20"],
        df["Price_Above_EMA50"],
        df["EMA20_Above_EMA50"],
        df["EMA20_Slope_Positive"],
        df["EMA20_EMA50_Cross_Up"]
    ]), axis=0)

    df["EMA_Bearish_Trend_Evidence"] = np.nanmean(np.vstack([
        1 - df["Price_Above_EMA20"],
        1 - df["Price_Above_EMA50"],
        1 - df["EMA20_Above_EMA50"],
        1 - df["EMA20_Slope_Positive"],
        df["EMA20_EMA50_Cross_Down"]
    ]), axis=0)

    df["EMA_Trend_Strength_Evidence"] = np.nanmean(np.vstack([
        df["EMA20_Slope"].abs().rank(pct=True),
        df["EMA50_Slope"].abs().rank(pct=True),
        df["Trend_Age_Bullish"].clip(0, 30) / 30.0,
        df["Trend_Age_Bearish"].clip(0, 30) / 30.0
    ]), axis=0)

    df["EMA_Overextension_Evidence"] = np.nanmean(np.vstack([
        df["EMA20_Overextended_Up"],
        df["EMA20_Overextended_Down"],
        df["EMA20_Distance_ATR"].abs().rank(pct=True)
    ]), axis=0)

    # =========================
    # MACD dots
    # =========================
    df["MACD_Above_Signal"] = (df["MACD"] > df["MACD_Signal"]).astype(float)
    df["MACD_Below_Signal"] = (df["MACD"] < df["MACD_Signal"]).astype(float)
    df["MACD_Cross_Up"] = ((df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))).astype(float)
    df["MACD_Cross_Down"] = ((df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))).astype(float)
    df["MACD_Above_Zero"] = (df["MACD"] > 0).astype(float)
    df["MACD_Below_Zero"] = (df["MACD"] < 0).astype(float)

    df["MACD_Hist_Positive"] = (df["MACD_Hist"] > 0).astype(float)
    df["MACD_Hist_Negative"] = (df["MACD_Hist"] < 0).astype(float)
    df["MACD_Hist_Rising"] = (df["MACD_Hist"].diff() > 0).astype(float)
    df["MACD_Hist_Falling"] = (df["MACD_Hist"].diff() < 0).astype(float)
    df["MACD_Hist_Zero_Cross_Up"] = ((df["MACD_Hist"] > 0) & (df["MACD_Hist"].shift(1) <= 0)).astype(float)
    df["MACD_Hist_Zero_Cross_Down"] = ((df["MACD_Hist"] < 0) & (df["MACD_Hist"].shift(1) >= 0)).astype(float)

    df["MACD_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["MACD_Above_Signal"],
        df["MACD_Cross_Up"],
        df["MACD_Above_Zero"],
        df["MACD_Hist_Positive"],
        df["MACD_Hist_Rising"]
    ]), axis=0)

    df["MACD_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["MACD_Below_Signal"],
        df["MACD_Cross_Down"],
        df["MACD_Below_Zero"],
        df["MACD_Hist_Negative"],
        df["MACD_Hist_Falling"]
    ]), axis=0)

    df["MACD_Momentum_Acceleration_Evidence"] = np.nanmean(np.vstack([
        df["MACD_Hist_Rising"],
        df["MACD_Hist_Zero_Cross_Up"]
    ]), axis=0)

    df["MACD_Momentum_Deceleration_Evidence"] = np.nanmean(np.vstack([
        df["MACD_Hist_Falling"],
        df["MACD_Hist_Zero_Cross_Down"]
    ]), axis=0)

    return df

def add_price_volume_structure_dots(df):
    df = df.copy()

    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    body = (df["Close"] - df["Open"]).abs()

    # =========================
    # Candle psychology dots
    # =========================
    df["Candle_Body_Pct"] = body / (rng + EPS)
    df["Candle_Upper_Wick_Pct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (rng + EPS)
    df["Candle_Lower_Wick_Pct"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (rng + EPS)
    df["Candle_Close_Position"] = (df["Close"] - df["Low"]) / (rng + EPS)

    df["Bullish_Candle"] = (df["Close"] > df["Open"]).astype(float)
    df["Bearish_Candle"] = (df["Close"] < df["Open"]).astype(float)
    df["Strong_Bullish_Candle"] = ((df["Bullish_Candle"] == 1) & (df["Candle_Body_Pct"] > 0.60) & (df["Candle_Close_Position"] > 0.70)).astype(float)
    df["Strong_Bearish_Candle"] = ((df["Bearish_Candle"] == 1) & (df["Candle_Body_Pct"] > 0.60) & (df["Candle_Close_Position"] < 0.30)).astype(float)
    df["Indecision_Candle"] = ((df["Candle_Body_Pct"] < 0.25) & (df["Candle_Upper_Wick_Pct"] > 0.25) & (df["Candle_Lower_Wick_Pct"] > 0.25)).astype(float)

    df["Upper_Wick_Rejection"] = ((df["Candle_Upper_Wick_Pct"] > 0.45) & (df["Candle_Close_Position"] < 0.55)).astype(float)
    df["Lower_Wick_Rejection"] = ((df["Candle_Lower_Wick_Pct"] > 0.45) & (df["Candle_Close_Position"] > 0.45)).astype(float)
    df["Buyer_Control_Candle"] = (df["Candle_Close_Position"] > 0.70).astype(float)
    df["Seller_Control_Candle"] = (df["Candle_Close_Position"] < 0.30).astype(float)

    df["Wide_Range_Candle"] = (rng > rng.rolling(20).quantile(0.80)).astype(float)
    df["Narrow_Range_Candle"] = (rng < rng.rolling(20).quantile(0.20)).astype(float)
    df["Inside_Bar_Dot"] = ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).astype(float)
    df["Outside_Bar_Dot"] = ((df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))).astype(float)

    df["Bullish_Candle_Streak_Dot"] = _consecutive_condition_count(df["Close"] > df["Open"])
    df["Bearish_Candle_Streak_Dot"] = _consecutive_condition_count(df["Close"] < df["Open"])

    df["Candle_Buyer_Control_Evidence"] = np.nanmean(np.vstack([
        df["Strong_Bullish_Candle"],
        df["Buyer_Control_Candle"],
        df["Lower_Wick_Rejection"],
        df["Bullish_Candle_Streak_Dot"].clip(0, 5) / 5.0
    ]), axis=0)

    df["Candle_Seller_Control_Evidence"] = np.nanmean(np.vstack([
        df["Strong_Bearish_Candle"],
        df["Seller_Control_Candle"],
        df["Upper_Wick_Rejection"],
        df["Bearish_Candle_Streak_Dot"].clip(0, 5) / 5.0
    ]), axis=0)

    df["Candle_Rejection_Evidence"] = np.nanmean(np.vstack([
        df["Upper_Wick_Rejection"],
        df["Lower_Wick_Rejection"],
        df["Indecision_Candle"]
    ]), axis=0)

    df["Candle_Indecision_Evidence"] = np.nanmean(np.vstack([
        df["Indecision_Candle"],
        df["Inside_Bar_Dot"],
        df["Narrow_Range_Candle"]
    ]), axis=0)

    df["Candle_Breakout_Evidence"] = np.nanmean(np.vstack([
        df["Wide_Range_Candle"],
        df["Strong_Bullish_Candle"],
        df["Strong_Bearish_Candle"],
        df["Candle_Body_Pct"]
    ]), axis=0)

    # =========================
    # Volume / flow dots
    # =========================
    vol_ma20 = df["Volume"].rolling(20).mean()
    df["Volume_Above_20D_Avg"] = (df["Volume"] > vol_ma20).astype(float)
    df["Volume_Expansion_Dot"] = (df["Volume"] > 1.5 * vol_ma20).astype(float)
    df["Volume_Dryup_Dot"] = (df["Volume"] < 0.7 * vol_ma20).astype(float)
    df["Volume_Percentile_High"] = (_rolling_percentile(df["Volume"], 60) > 0.80).astype(float)
    df["Volume_Percentile_Low"] = (_rolling_percentile(df["Volume"], 60) < 0.20).astype(float)

    df["OBV_Slope_Positive"] = (df["OBV_Slope"] > 0).astype(float)
    df["OBV_Slope_Negative"] = (df["OBV_Slope"] < 0).astype(float)
    df["CMF_Positive_Dot"] = (df["CMF20"] > 0).astype(float)
    df["CMF_Negative_Dot"] = (df["CMF20"] < 0).astype(float)
    df["MFI_Above_50"] = (df["MFI14"] > 50).astype(float)
    df["MFI_Below_50"] = (df["MFI14"] < 50).astype(float)

    df["Volume_Delta_Positive_Dot"] = (df["Volume_Delta"] > 0).astype(float)
    df["Volume_Delta_Negative_Dot"] = (df["Volume_Delta"] < 0).astype(float)
    df["UpDown_Volume_Ratio_Strong"] = (df["UpDown_Volume_Ratio"] > 1.25).astype(float)
    df["UpDown_Volume_Ratio_Weak"] = (df["UpDown_Volume_Ratio"] < 0.80).astype(float)

    df["Volume_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["OBV_Slope_Positive"],
        df["CMF_Positive_Dot"],
        df["MFI_Above_50"],
        df["Volume_Delta_Positive_Dot"],
        df["UpDown_Volume_Ratio_Strong"]
    ]), axis=0)

    df["Volume_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["OBV_Slope_Negative"],
        df["CMF_Negative_Dot"],
        df["MFI_Below_50"],
        df["Volume_Delta_Negative_Dot"],
        df["UpDown_Volume_Ratio_Weak"]
    ]), axis=0)

    df["Volume_Confirmation_Evidence"] = np.nanmean(np.vstack([
        df["Volume_Above_20D_Avg"],
        df["Volume_Expansion_Dot"],
        df["Volume_Percentile_High"]
    ]), axis=0)

    df["Volume_Divergence_Evidence"] = np.nanmean(np.vstack([
        (df["Close"].pct_change(5) > 0).astype(float) * df["OBV_Slope_Negative"],
        (df["Close"].pct_change(5) < 0).astype(float) * df["OBV_Slope_Positive"]
    ]), axis=0)

    df["Liquidity_Shock_Evidence"] = _rolling_z(df["Close"] * df["Volume"], 20).rank(pct=True)

    # =========================
    # Volatility / compression dots
    # =========================
    df["ATR_Pct_High"] = (_rolling_percentile(df["ATR_Pct"], 60) > 0.80).astype(float)
    df["ATR_Pct_Low"] = (_rolling_percentile(df["ATR_Pct"], 60) < 0.20).astype(float)
    df["Volatility_Compression_Dot"] = (df["Volatility_Compression_20"] < 0.80).astype(float)
    df["Volatility_Expansion_Dot"] = (df["Volatility_Compression_20"] > 1.20).astype(float)

    df["BB_Squeeze_Active"] = (df["BB_Squeeze"] > 0).astype(float)
    df["BB_Expansion_Active"] = (df["BB_Width"] > df["BB_Width"].rolling(60).quantile(0.80)).astype(float)
    df["Range_Compression_Dot"] = (df["Range_Compression_5_20"] < 0.80).astype(float)
    df["Range_Expansion_Dot"] = (df["Range_Compression_5_20"] > 1.20).astype(float)

    df["Compression_Building"] = np.nanmean(np.vstack([
        df["BB_Squeeze_Active"],
        df["Range_Compression_Dot"],
        df["Volatility_Compression_Dot"],
        df["Volume_Dryup_Dot"]
    ]), axis=0)

    df["Compression_Release"] = np.nanmean(np.vstack([
        df["BB_Expansion_Active"],
        df["Range_Expansion_Dot"],
        df["Volume_Expansion_Dot"],
        df["Wide_Range_Candle"]
    ]), axis=0)

    df["Squeeze_Intensity_High"] = (df["Squeeze_Intensity"] > df["Squeeze_Intensity"].rolling(60).quantile(0.80)).astype(float)

    df["Volatility_Expansion_Evidence"] = np.nanmean(np.vstack([
        df["Volatility_Expansion_Dot"],
        df["BB_Expansion_Active"],
        df["Range_Expansion_Dot"],
        df["Compression_Release"]
    ]), axis=0)

    df["Compression_Breakout_Evidence"] = np.nanmean(np.vstack([
        df["Compression_Building"],
        df["Squeeze_Intensity_High"],
        df["BreakReliab"].rank(pct=True)
    ]), axis=0)

    df["High_Risk_Volatility_Evidence"] = np.nanmean(np.vstack([
        df["ATR_Pct_High"],
        df["Volatility_Expansion_Dot"],
        df["Crash_Risk_Score"].rank(pct=True)
    ]), axis=0)

    df["Low_Noise_Compression_Evidence"] = np.nanmean(np.vstack([
        df["ATR_Pct_Low"],
        df["Compression_Building"],
        1 - df["Choppiness_Index"].rank(pct=True)
    ]), axis=0)

    # =========================
    # Structure dots
    # =========================
    df["Near_20D_High"] = (df["Distance_From_20D_High"].abs() < 0.01).astype(float)
    df["Near_20D_Low"] = (df["Distance_From_20D_Low"].abs() < 0.01).astype(float)
    df["Near_52W_High"] = (df["Distance_From_52W_High"].abs() < 0.03).astype(float)
    df["Near_52W_Low"] = (df["Distance_From_52W_Low"].abs() < 0.03).astype(float)

    df["Donchian_Upper_Break"] = (df["Breakout_Distance"] > 0).astype(float)
    df["Donchian_Lower_Break"] = (df["Close"] < df["Donchian_Low20"].shift(1)).astype(float)
    df["Donchian_Mid_Above"] = (df["Donchian_Pos"] > 0.50).astype(float)
    df["Donchian_Mid_Below"] = (df["Donchian_Pos"] < 0.50).astype(float)

    df["Breakout_Distance_Positive"] = (df["Breakout_Distance"] > 0).astype(float)
    df["Breakdown_Distance_Negative"] = df["Donchian_Lower_Break"]
    df["Pullback_Depth_High"] = (_rolling_percentile(df["Pullback_Depth"], 60) > 0.80).astype(float)

    df["Structure_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["Donchian_Mid_Above"],
        df["Donchian_Upper_Break"],
        df["Near_20D_High"],
        df["Breakout_Distance_Positive"]
    ]), axis=0)

    df["Structure_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["Donchian_Mid_Below"],
        df["Donchian_Lower_Break"],
        df["Near_20D_Low"],
        df["Breakdown_Distance_Negative"]
    ]), axis=0)

    df["Breakout_Evidence"] = np.nanmean(np.vstack([
        df["Donchian_Upper_Break"],
        df["Breakout_Distance_Positive"],
        df["Candle_Breakout_Evidence"],
        df["Near_20D_High"]
    ]), axis=0)

    df["Breakdown_Evidence"] = np.nanmean(np.vstack([
        df["Donchian_Lower_Break"],
        df["Breakdown_Distance_Negative"],
        df["Strong_Bearish_Candle"],
        df["Near_20D_Low"]
    ]), axis=0)

    df["Pullback_Evidence"] = np.nanmean(np.vstack([
        df["Pullback_Depth_High"],
        df["Near_20D_Low"],
        df["Lower_Wick_Rejection"]
    ]), axis=0)

    df["Support_Reaction_Evidence"] = np.nanmean(np.vstack([
        df["Near_20D_Low"],
        df["Lower_Wick_Rejection"],
        df["RSI_Exit_Oversold"]
    ]), axis=0)

    df["Resistance_Rejection_Evidence"] = np.nanmean(np.vstack([
        df["Near_20D_High"],
        df["Upper_Wick_Rejection"],
        df["RSI_Exit_Overbought"]
    ]), axis=0)

    # =========================
    # Relative strength dots
    # =========================
    df["RS_Nifty_Positive"] = (df["RS_Nifty_20D"] > 0).astype(float)
    df["RS_Nifty_Negative"] = (df["RS_Nifty_20D"] < 0).astype(float)
    df["RS_Sector_Positive"] = (df["RS_Sector_20D"] > 0).astype(float)
    df["RS_Sector_Negative"] = (df["RS_Sector_20D"] < 0).astype(float)

    df["RS_Nifty_Improving"] = (df["RS_Nifty_20D"].diff(5) > 0).astype(float)
    df["RS_Nifty_Weakening"] = (df["RS_Nifty_20D"].diff(5) < 0).astype(float)
    df["RS_Sector_Improving"] = (df["RS_Sector_20D"].diff(5) > 0).astype(float)
    df["RS_Sector_Weakening"] = (df["RS_Sector_20D"].diff(5) < 0).astype(float)

    df["Alpha_Positive_Dot"] = (df["Rolling_Alpha_20"] > 0).astype(float)
    df["Alpha_Negative_Dot"] = (df["Rolling_Alpha_20"] < 0).astype(float)
    df["Alpha_Persistence_Strong"] = (df["Alpha_Persistence_20"] > 5).astype(float)

    df["RelativeStrength_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["RS_Nifty_Positive"],
        df["RS_Sector_Positive"],
        df["RS_Nifty_Improving"],
        df["RS_Sector_Improving"],
        df["Alpha_Positive_Dot"]
    ]), axis=0)

    df["RelativeStrength_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["RS_Nifty_Negative"],
        df["RS_Sector_Negative"],
        df["RS_Nifty_Weakening"],
        df["RS_Sector_Weakening"],
        df["Alpha_Negative_Dot"]
    ]), axis=0)

    df["Leadership_Evidence"] = np.nanmean(np.vstack([
        df["RelativeStrength_Bullish_Evidence"],
        df["Alpha_Persistence_Strong"],
        df["RS_Persistence_20"].clip(0, 10) / 10.0
    ]), axis=0)

    df["Weakness_Evidence"] = df["RelativeStrength_Bearish_Evidence"]

    # =========================
    # Gap dots
    # =========================
    df["Gap_Up_Dot"] = (df["Gap_vs_ATR"] > 0).astype(float)
    df["Gap_Down_Dot"] = (df["Gap_vs_ATR"] < 0).astype(float)
    df["Large_Gap_Up"] = (df["Gap_vs_ATR"] > 1.0).astype(float)
    df["Large_Gap_Down"] = (df["Gap_vs_ATR"] < -1.0).astype(float)

    df["Gap_Filled_Dot"] = (df["Gap_Fill_Ratio"] > 0.5).astype(float)
    df["Gap_Not_Filled_Dot"] = 1 - df["Gap_Filled_Dot"]
    df["Gap_Continuation_Bullish"] = ((df["Gap_Up_Dot"] == 1) & (df["Bullish_Candle"] == 1) & (df["Candle_Close_Position"] > 0.60)).astype(float)
    df["Gap_Continuation_Bearish"] = ((df["Gap_Down_Dot"] == 1) & (df["Bearish_Candle"] == 1) & (df["Candle_Close_Position"] < 0.40)).astype(float)

    df["Gap_Exhaustion_Dot"] = (df["Gap_Exhaustion_Score"] > df["Gap_Exhaustion_Score"].rolling(60).quantile(0.80)).astype(float)

    df["Gap_Bullish_Evidence"] = np.nanmean(np.vstack([
        df["Gap_Up_Dot"],
        df["Gap_Continuation_Bullish"],
        df["Opening_Gap_Strength"].rank(pct=True)
    ]), axis=0)

    df["Gap_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["Gap_Down_Dot"],
        df["Gap_Continuation_Bearish"],
        (-df["Opening_Gap_Strength"]).rank(pct=True)
    ]), axis=0)

    df["Gap_Rejection_Evidence"] = np.nanmean(np.vstack([
        df["Gap_Filled_Dot"],
        df["Gap_Exhaustion_Dot"],
        df["Upper_Wick_Rejection"],
        df["Lower_Wick_Rejection"]
    ]), axis=0)

    # =========================
    # Risk / exhaustion dots
    # =========================
    df["Overextension_High"] = (df["Overextension_Score"] > df["Overextension_Score"].rolling(60).quantile(0.80)).astype(float)
    df["Distance_From_EMA20_ATR_High"] = (df["Distance_From_EMA20_ATR"].abs() > 2.0).astype(float)
    df["Distance_From_EMA50_ATR_High"] = (df["Distance_From_EMA50_ATR"].abs() > 3.0).astype(float)

    df["Consecutive_Up_Days_High"] = (df["Consecutive_Up_Days"] >= 5).astype(float)
    df["Consecutive_Down_Days_High"] = (df["Consecutive_Down_Days"] >= 5).astype(float)

    df["Crash_Risk_High"] = (df["Crash_Risk_Score"] > df["Crash_Risk_Score"].rolling(60).quantile(0.80)).astype(float)
    df["Drawdown_Speed_High"] = (df["Drawdown_Speed"] < df["Drawdown_Speed"].rolling(60).quantile(0.20)).astype(float)
    df["Left_Tail_Risk_High"] = (df["Left_Tail_Return_Count_20"] > df["Left_Tail_Return_Count_20"].rolling(60).quantile(0.80)).astype(float)
    df["Exhaustion_Risk_High"] = np.nanmean(np.vstack([
        df["Overextension_High"],
        df["Distance_From_EMA20_ATR_High"],
        df["RSI_Exhaustion_Evidence"],
        df["Consecutive_Up_Days_High"]
    ]), axis=0)

    df["Risk_Bearish_Evidence"] = np.nanmean(np.vstack([
        df["Crash_Risk_High"],
        df["Drawdown_Speed_High"],
        df["Left_Tail_Risk_High"],
        df["High_Risk_Volatility_Evidence"]
    ]), axis=0)

    df["Risk_Reversal_Evidence"] = np.nanmean(np.vstack([
        df["Exhaustion_Risk_High"],
        df["Candle_Rejection_Evidence"],
        df["RSI_Exit_Overbought"],
        df["Resistance_Rejection_Evidence"]
    ]), axis=0)

    df["Trend_Maturity_Evidence"] = np.nanmean(np.vstack([
        df["Mature_Bullish_Trend"],
        df["Mature_Bearish_Trend"],
        df["Trend_Age_Bullish"].clip(0, 30) / 30.0,
        df["Trend_Age_Bearish"].clip(0, 30) / 30.0
    ]), axis=0)

    df["No_Trade_Risk_Evidence"] = np.nanmean(np.vstack([
        df["Risk_Bearish_Evidence"],
        df["Candle_Indecision_Evidence"],
        df["RSI_Neutral_45_55"],
        df["Choppiness_Index"].rank(pct=True)
    ]), axis=0)

    return df

def add_cross_family_dot_connections(df):
    df = df.copy()

    # RSI × EMA
    df["RSI_Trend_Bullish_Agreement"] = df["RSI_Bullish_Evidence"] * df["EMA_Bullish_Trend_Evidence"]
    df["RSI_Trend_Bearish_Agreement"] = df["RSI_Bearish_Evidence"] * df["EMA_Bearish_Trend_Evidence"]
    df["RSI_Trend_Conflict"] = (
        df["RSI_Bullish_Evidence"] * df["EMA_Bearish_Trend_Evidence"]
        + df["RSI_Bearish_Evidence"] * df["EMA_Bullish_Trend_Evidence"]
    ) / 2.0
    df["RSI_Overbought_Trend_Strength"] = df["RSI_Overbought_70"] * df["EMA_Bullish_Trend_Evidence"] * df["Volume_Confirmation_Evidence"]
    df["RSI_Overbought_Exhaustion_Risk"] = df["RSI_Overbought_70"] * df["EMA_Overextension_Evidence"] * df["Candle_Rejection_Evidence"]
    df["RSI_Oversold_Reversal_Setup"] = df["RSI_Oversold_30"] * df["Support_Reaction_Evidence"] * df["Candle_Buyer_Control_Evidence"]

    # RSI × Candle
    df["RSI_Candle_Bullish_Confirmation"] = df["RSI_Bullish_Evidence"] * df["Candle_Buyer_Control_Evidence"]
    df["RSI_Candle_Bearish_Confirmation"] = df["RSI_Bearish_Evidence"] * df["Candle_Seller_Control_Evidence"]
    df["RSI_Candle_Reversal_Evidence"] = (
        df["RSI_Exit_Oversold"] * df["Lower_Wick_Rejection"]
        + df["RSI_Exit_Overbought"] * df["Upper_Wick_Rejection"]
    )
    df["RSI_Exit_Oversold_Bullish_Candle"] = df["RSI_Exit_Oversold"] * df["Strong_Bullish_Candle"]
    df["RSI_Exit_Overbought_Bearish_Candle"] = df["RSI_Exit_Overbought"] * df["Strong_Bearish_Candle"]

    # RSI × Volume
    df["RSI_Volume_Bullish_Confirmation"] = df["RSI_Bullish_Evidence"] * df["Volume_Bullish_Evidence"]
    df["RSI_Volume_Bearish_Confirmation"] = df["RSI_Bearish_Evidence"] * df["Volume_Bearish_Evidence"]
    df["RSI_Activation_With_Volume"] = (
        df["RSI_Escape_Above_55"] + df["RSI_Escape_Below_45"]
    ).clip(0, 1) * df["Volume_Confirmation_Evidence"]
    df["RSI_Activation_Without_Volume_Risk"] = (
        df["RSI_Escape_Above_55"] + df["RSI_Escape_Below_45"]
    ).clip(0, 1) * (1 - df["Volume_Confirmation_Evidence"])

    # EMA × Volume
    df["Trend_Volume_Bullish_Confirmation"] = df["EMA_Bullish_Trend_Evidence"] * df["Volume_Bullish_Evidence"]
    df["Trend_Volume_Bearish_Confirmation"] = df["EMA_Bearish_Trend_Evidence"] * df["Volume_Bearish_Evidence"]
    df["Trend_Without_Volume_Risk"] = (
        df["EMA_Bullish_Trend_Evidence"] + df["EMA_Bearish_Trend_Evidence"]
    ).clip(0, 1) * (1 - df["Volume_Confirmation_Evidence"])
    df["Volume_Against_Trend_Warning"] = (
        df["EMA_Bullish_Trend_Evidence"] * df["Volume_Bearish_Evidence"]
        + df["EMA_Bearish_Trend_Evidence"] * df["Volume_Bullish_Evidence"]
    ) / 2.0

    # EMA × Candle
    df["Trend_Candle_Bullish_Confirmation"] = df["EMA_Bullish_Trend_Evidence"] * df["Candle_Buyer_Control_Evidence"]
    df["Trend_Candle_Bearish_Confirmation"] = df["EMA_Bearish_Trend_Evidence"] * df["Candle_Seller_Control_Evidence"]
    df["Trend_Candle_Rejection_Warning"] = (
        df["EMA_Bullish_Trend_Evidence"] * df["Upper_Wick_Rejection"]
        + df["EMA_Bearish_Trend_Evidence"] * df["Lower_Wick_Rejection"]
    ) / 2.0
    df["Trend_Candle_Pullback_Opportunity"] = (
        df["EMA_Bullish_Trend_Evidence"] * df["Pullback_Evidence"] * df["Lower_Wick_Rejection"]
        + df["EMA_Bearish_Trend_Evidence"] * df["Pullback_Evidence"] * df["Upper_Wick_Rejection"]
    ) / 2.0

    # Breakout × Volume
    df["Breakout_Volume_Confirmation"] = df["Breakout_Evidence"] * df["Volume_Confirmation_Evidence"] * df["Candle_Buyer_Control_Evidence"]
    df["Breakout_Without_Volume_Risk"] = df["Breakout_Evidence"] * (1 - df["Volume_Confirmation_Evidence"])
    df["Breakout_Failure_Risk"] = df["Breakout_Evidence"] * df["Upper_Wick_Rejection"] * (1 - df["Volume_Confirmation_Evidence"])
    df["False_Breakout_Evidence"] = np.nanmean(np.vstack([
        df["Breakout_Failure_Risk"],
        df["Resistance_Rejection_Evidence"],
        df["Candle_Rejection_Evidence"]
    ]), axis=0)

    # Breakdown × Volume
    df["Breakdown_Volume_Confirmation"] = df["Breakdown_Evidence"] * df["Volume_Confirmation_Evidence"] * df["Candle_Seller_Control_Evidence"]
    df["Breakdown_Without_Volume_Risk"] = df["Breakdown_Evidence"] * (1 - df["Volume_Confirmation_Evidence"])

    # Compression × Breakout
    df["Compression_Breakout_Readiness"] = df["Compression_Building"] * df["Breakout_Evidence"] * df["Volume_Confirmation_Evidence"]
    df["Squeeze_Release_Evidence"] = df["Squeeze_Intensity_High"] * df["Compression_Release"]
    df["Volatility_Expansion_Setup"] = df["Compression_Building"] * df["Volatility_Expansion_Evidence"]
    df["Post_Compression_Failure_Risk"] = df["Compression_Building"] * df["Candle_Rejection_Evidence"] * (1 - df["Volume_Confirmation_Evidence"])

    # Trend × Relative Strength
    df["Trend_RS_Leadership"] = df["EMA_Bullish_Trend_Evidence"] * df["Leadership_Evidence"]
    df["Trend_RS_Weakness"] = df["EMA_Bullish_Trend_Evidence"] * df["Weakness_Evidence"]
    df["Bullish_Trend_With_Sector_Leadership"] = df["EMA_Bullish_Trend_Evidence"] * df["RS_Sector_Positive"]
    df["Bearish_Trend_With_Sector_Weakness"] = df["EMA_Bearish_Trend_Evidence"] * df["RS_Sector_Negative"]

    # Risk × Trend
    df["Trend_Exhaustion_Risk"] = df["EMA_Bullish_Trend_Evidence"] * df["EMA_Overextension_Evidence"] * df["RSI_Exhaustion_Evidence"]
    df["Trend_Continuation_Quality"] = df["EMA_Bullish_Trend_Evidence"] * df["RSI_Bullish_Evidence"] * df["Volume_Bullish_Evidence"] * (1 - df["Risk_Reversal_Evidence"])
    df["Overextension_Reversal_Risk"] = df["EMA_Overextension_Evidence"] * df["Candle_Rejection_Evidence"] * df["RSI_Exhaustion_Evidence"]
    df["Healthy_Trend_Pullback"] = df["EMA_Bullish_Trend_Evidence"] * df["Pullback_Evidence"] * (1 - df["EMA_Overextension_Evidence"])

    # Gap × Candle × Volume
    df["Gap_Candle_Continuation_Confirmation"] = (
        df["Gap_Continuation_Bullish"] * df["Candle_Buyer_Control_Evidence"]
        + df["Gap_Continuation_Bearish"] * df["Candle_Seller_Control_Evidence"]
    )
    df["Gap_Candle_Rejection_Warning"] = df["Gap_Exhaustion_Dot"] * df["Candle_Rejection_Evidence"]
    df["Gap_Volume_Continuation_Evidence"] = (
        df["Gap_Continuation_Bullish"] + df["Gap_Continuation_Bearish"]
    ).clip(0, 1) * df["Volume_Confirmation_Evidence"]
    df["Gap_Exhaustion_With_Weak_Close"] = df["Gap_Exhaustion_Dot"] * df["Indecision_Candle"] * (1 - df["Volume_Confirmation_Evidence"])

    # MACD × RSI × EMA
    df["Momentum_Stack_Bullish"] = df["RSI_Bullish_Evidence"] * df["MACD_Bullish_Evidence"] * df["EMA_Bullish_Trend_Evidence"]
    df["Momentum_Stack_Bearish"] = df["RSI_Bearish_Evidence"] * df["MACD_Bearish_Evidence"] * df["EMA_Bearish_Trend_Evidence"]
    df["Momentum_Trend_Disagreement"] = (
        df["MACD_Bullish_Evidence"] * df["EMA_Bearish_Trend_Evidence"]
        + df["MACD_Bearish_Evidence"] * df["EMA_Bullish_Trend_Evidence"]
    ) / 2.0

    return df

def add_final_market_evidence_scores(df):
    df = df.copy()

    bullish_components = [
        "RSI_Bullish_Evidence",
        "MACD_Bullish_Evidence",
        "EMA_Bullish_Trend_Evidence",
        "Candle_Buyer_Control_Evidence",
        "Volume_Bullish_Evidence",
        "Structure_Bullish_Evidence",
        "RelativeStrength_Bullish_Evidence",
        "Trend_RS_Leadership",
        "Breakout_Volume_Confirmation",
        "Compression_Breakout_Readiness",
        "Momentum_Stack_Bullish",
        "Gap_Bullish_Evidence"
    ]

    bearish_components = [
        "RSI_Bearish_Evidence",
        "MACD_Bearish_Evidence",
        "EMA_Bearish_Trend_Evidence",
        "Candle_Seller_Control_Evidence",
        "Volume_Bearish_Evidence",
        "Structure_Bearish_Evidence",
        "RelativeStrength_Bearish_Evidence",
        "Breakdown_Volume_Confirmation",
        "Momentum_Stack_Bearish",
        "Gap_Bearish_Evidence",
        "Risk_Bearish_Evidence"
    ]

    reversal_components = [
        "RSI_Candle_Reversal_Evidence",
        "RSI_Oversold_Reversal_Setup",
        "Risk_Reversal_Evidence",
        "Overextension_Reversal_Risk",
        "Support_Reaction_Evidence",
        "Resistance_Rejection_Evidence",
        "Gap_Rejection_Evidence"
    ]

    breakout_components = [
        "Breakout_Evidence",
        "Breakout_Volume_Confirmation",
        "Compression_Breakout_Readiness",
        "Squeeze_Release_Evidence",
        "Volatility_Expansion_Setup",
        "Candle_Breakout_Evidence"
    ]

    noise_components = [
        "RSI_Neutral_45_55",
        "Candle_Indecision_Evidence",
        "Trend_Without_Volume_Risk",
        "Momentum_Trend_Disagreement",
        "RSI_Trend_Conflict",
        "No_Trade_Risk_Evidence"
    ]

    df["Bullish_Evidence_Total"] = np.nanmean(np.vstack([df[c] for c in bullish_components if c in df.columns]), axis=0)
    df["Bearish_Evidence_Total"] = np.nanmean(np.vstack([df[c] for c in bearish_components if c in df.columns]), axis=0)
    df["Net_Directional_Evidence"] = df["Bullish_Evidence_Total"] - df["Bearish_Evidence_Total"]

    df["Trend_Continuation_Evidence"] = np.nanmean(np.vstack([
        df["Trend_Continuation_Quality"],
        df["Trend_Volume_Bullish_Confirmation"],
        df["Trend_RS_Leadership"],
        df["Momentum_Stack_Bullish"]
    ]), axis=0)

    df["Reversal_Evidence"] = np.nanmean(np.vstack([df[c] for c in reversal_components if c in df.columns]), axis=0)
    df["Breakout_Readiness_Evidence"] = np.nanmean(np.vstack([df[c] for c in breakout_components if c in df.columns]), axis=0)
    df["Breakdown_Readiness_Evidence"] = np.nanmean(np.vstack([
        df["Breakdown_Evidence"],
        df["Breakdown_Volume_Confirmation"],
        df["Momentum_Stack_Bearish"],
        df["Candle_Seller_Control_Evidence"]
    ]), axis=0)

    df["False_Breakout_Risk"] = np.nanmean(np.vstack([
        df["False_Breakout_Evidence"],
        df["Breakout_Without_Volume_Risk"],
        df["Post_Compression_Failure_Risk"],
        df["Resistance_Rejection_Evidence"]
    ]), axis=0)

    df["Volatility_Expansion_Evidence_Final"] = np.nanmean(np.vstack([
        df["Volatility_Expansion_Evidence"],
        df["Compression_Release"],
        df["Squeeze_Release_Evidence"],
        df["Range_Expansion_Dot"]
    ]), axis=0)

    df["Exhaustion_Risk_Evidence"] = np.nanmean(np.vstack([
        df["RSI_Exhaustion_Evidence"],
        df["EMA_Overextension_Evidence"],
        df["Trend_Exhaustion_Risk"],
        df["Overextension_Reversal_Risk"],
        df["Gap_Exhaustion_Dot"]
    ]), axis=0)

    df["Trade_Quality_Evidence"] = (
        df["Bullish_Evidence_Total"].abs()
        + df["Bearish_Evidence_Total"].abs()
        + df["Breakout_Readiness_Evidence"]
        + df["Trend_Continuation_Evidence"]
        - df["No_Trade_Risk_Evidence"]
        - df["False_Breakout_Risk"]
    )

    df["No_Trade_Noise_Evidence"] = np.nanmean(np.vstack([df[c] for c in noise_components if c in df.columns]), axis=0)

    return df

def add_probability_ecosystem_features(df):
    df = df.copy()

    bias_bin = pd.cut(
        df["Ecosystem_Directional_Bias"].fillna(0),
        bins=[-np.inf, -0.35, 0.35, np.inf],
        labels=[0, 1, 2]
    ).astype(float).fillna(1).astype(int)

    vol_bin = pd.cut(
        df["Volatility_Score"].fillna(0),
        bins=[-np.inf, -0.25, 0.25, np.inf],
        labels=[0, 1, 2]
    ).astype(float).fillna(1).astype(int)

    compression_bin = pd.cut(
        df["Compression_Score"].fillna(0),
        bins=[-np.inf, -0.25, 0.25, np.inf],
        labels=[0, 1, 2]
    ).astype(float).fillna(1).astype(int)

    risk_bin = pd.cut(
        df["RiskDrawdown_Score"].fillna(0),
        bins=[-np.inf, -0.25, 0.25, np.inf],
        labels=[0, 1, 2]
    ).astype(float).fillna(1).astype(int)

    df["Ecosystem_State_ID"] = (
        bias_bin * 27
        + vol_bin * 9
        + compression_bin * 3
        + risk_bin
    ).astype(float)

    future_ret = df["LogRet"].shift(-1)
    states = df["Ecosystem_State_ID"].astype(int).values
    n = len(df)

    p_up = np.full(n, np.nan)
    p_down = np.full(n, np.nan)
    p_flat = np.full(n, np.nan)
    exp_ret = np.full(n, np.nan)
    exp_abs = np.full(n, np.nan)
    exp_down = np.full(n, np.nan)
    exp_up = np.full(n, np.nan)
    odds = np.full(n, np.nan)
    payoff = np.full(n, np.nan)
    edge = np.full(n, np.nan)
    uncert = np.full(n, np.nan)
    freq = np.full(n, np.nan)
    rarity = np.full(n, np.nan)
    sample_conf = np.full(n, np.nan)

    state_stats = {}
    total_seen = 0

    transition_stats = {}
    prob_cont = np.full(n, np.nan)
    prob_rev = np.full(n, np.nan)
    prob_breakout = np.full(n, np.nan)
    prob_breakdown = np.full(n, np.nan)
    prob_meanrev = np.full(n, np.nan)
    prob_vol_exp = np.full(n, np.nan)
    prob_vol_comp = np.full(n, np.nan)

    for i in range(n):
        # update state statistics using previous row's realized next-day return
        if i > 0:
            prev_state = int(states[i - 1])
            r = future_ret.iloc[i - 1]
            if not pd.isna(r):
                if prev_state not in state_stats:
                    state_stats[prev_state] = {
                        "count": 0, "up": 0, "down": 0, "flat": 0,
                        "sum_ret": 0.0, "sum_abs": 0.0, "sum_up": 0.0, "sum_down": 0.0,
                        "up_count": 0, "down_count": 0
                    }
                st = state_stats[prev_state]
                st["count"] += 1
                st["sum_ret"] += float(r)
                st["sum_abs"] += abs(float(r))
                if r > 0.001:
                    st["up"] += 1
                    st["sum_up"] += float(r)
                    st["up_count"] += 1
                elif r < -0.001:
                    st["down"] += 1
                    st["sum_down"] += float(r)
                    st["down_count"] += 1
                else:
                    st["flat"] += 1
                total_seen += 1

        # update transition statistics using completed previous transition
        if i > 1:
            s_from = int(states[i - 2])
            s_to = int(states[i - 1])
            if s_from not in transition_stats:
                transition_stats[s_from] = {}
            transition_stats[s_from][s_to] = transition_stats[s_from].get(s_to, 0) + 1

        cur_state = int(states[i])
        st = state_stats.get(cur_state)

        if st and st["count"] > 0:
            c = st["count"]
            # Bayesian smoothing with mild 1/3 prior
            p_up[i] = (st["up"] + 1.0) / (c + 3.0)
            p_down[i] = (st["down"] + 1.0) / (c + 3.0)
            p_flat[i] = (st["flat"] + 1.0) / (c + 3.0)

            exp_ret[i] = st["sum_ret"] / c
            exp_abs[i] = st["sum_abs"] / c
            exp_up[i] = st["sum_up"] / (st["up_count"] + EPS)
            exp_down[i] = st["sum_down"] / (st["down_count"] + EPS)

            odds[i] = p_up[i] / (p_down[i] + EPS)
            payoff[i] = exp_up[i] / (abs(exp_down[i]) + EPS)
            edge[i] = p_up[i] * exp_up[i] - p_down[i] * abs(exp_down[i])
            uncert[i] = 1.0 - max(p_up[i], p_down[i], p_flat[i])

            freq[i] = c / (total_seen + EPS)
            rarity[i] = 1.0 - freq[i]
            sample_conf[i] = c / (c + 20.0)
        else:
            p_up[i] = p_down[i] = p_flat[i] = 1.0 / 3.0
            exp_ret[i] = exp_abs[i] = exp_up[i] = exp_down[i] = 0.0
            odds[i] = payoff[i] = 1.0
            edge[i] = 0.0
            uncert[i] = 1.0
            freq[i] = 0.0
            rarity[i] = 1.0
            sample_conf[i] = 0.0

        trans = transition_stats.get(cur_state, {})
        trans_total = sum(trans.values())
        if trans_total > 0:
            prob_cont[i] = trans.get(cur_state, 0) / trans_total
            prob_rev[i] = 1.0 - prob_cont[i]

            # Approximate transition categories from destination state's encoded bins
            breakout_count = 0
            breakdown_count = 0
            meanrev_count = 0
            vol_exp_count = 0
            vol_comp_count = 0

            for dest_state, cnt in trans.items():
                dest_bias = dest_state // 27
                dest_vol = (dest_state % 27) // 9
                dest_comp = (dest_state % 9) // 3

                if dest_bias == 2 and dest_comp >= 1:
                    breakout_count += cnt
                if dest_bias == 0 and dest_comp >= 1:
                    breakdown_count += cnt
                if dest_bias == 1:
                    meanrev_count += cnt
                if dest_vol == 2:
                    vol_exp_count += cnt
                if dest_vol == 0:
                    vol_comp_count += cnt

            prob_breakout[i] = breakout_count / trans_total
            prob_breakdown[i] = breakdown_count / trans_total
            prob_meanrev[i] = meanrev_count / trans_total
            prob_vol_exp[i] = vol_exp_count / trans_total
            prob_vol_comp[i] = vol_comp_count / trans_total
        else:
            prob_cont[i] = prob_rev[i] = 0.5
            prob_breakout[i] = prob_breakdown[i] = prob_meanrev[i] = 1.0 / 3.0
            prob_vol_exp[i] = prob_vol_comp[i] = 0.5

    df["Ecosystem_State_Frequency"] = freq
    df["Ecosystem_State_Rarity"] = rarity

    df["P_Up_Given_Ecosystem_State"] = p_up
    df["P_Down_Given_Ecosystem_State"] = p_down
    df["P_Flat_Given_Ecosystem_State"] = p_flat

    df["Expected_Return_Given_Ecosystem_State"] = exp_ret
    df["Expected_AbsReturn_Given_Ecosystem_State"] = exp_abs
    df["Expected_Downside_Given_Ecosystem_State"] = exp_down
    df["Expected_Upside_Given_Ecosystem_State"] = exp_up

    df["WinLoss_Odds_Given_Ecosystem_State"] = odds
    df["Payoff_Ratio_Given_Ecosystem_State"] = payoff
    df["Ecosystem_Edge"] = edge
    df["Ecosystem_Uncertainty"] = uncert

    df["Prob_State_Continuation"] = prob_cont
    df["Prob_State_Reversal"] = prob_rev
    df["Prob_Breakout_Transition"] = prob_breakout
    df["Prob_Breakdown_Transition"] = prob_breakdown
    df["Prob_MeanReversion_Transition"] = prob_meanrev
    df["Prob_Volatility_Expansion"] = prob_vol_exp
    df["Prob_Volatility_Compression"] = prob_vol_comp

    df["State_Sample_Confidence"] = sample_conf
    df["State_Probability_Stability"] = 1.0 / (_rolling_z(df["P_Up_Given_Ecosystem_State"], 60).abs() + 1.0)
    df["State_Edge_Stability"] = 1.0 / (df["Ecosystem_Edge"].rolling(60).std() + EPS)

    return df

# ---------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------
def build_features_from_df(df_raw):
    df = df_raw.copy()

    rename_dict = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    for csv_col, internal_col in SECTOR_INDEX_INTERNAL.items():
        if csv_col in df.columns and internal_col not in df.columns:
            df.rename(columns={csv_col: internal_col}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    num_cols = ["Close", "High", "Low", "Open", "Volume", "Change %",
                "^NSEI", "^BSESN"] + list(SECTOR_INDEX_INTERNAL.values())
    df = clean_numeric_cols(df, num_cols)

    fundamental_cols = [
        "pe_ratio", "pb_ratio", "roe", "roa", "debt_to_equity",
        "current_ratio", "quick_ratio", "profit_margins", "operating_margins",
        "ebitda_margins", "revenue_growth"
    ]
    df = clean_numeric_cols(df, fundamental_cols)

    if "Change %" not in df.columns or df["Change %"].isna().all():
        df["Change %"] = df["Close"].pct_change() * 100

    df["RSI14"]   = rsi(df["Close"], 14)
    df["EMA5"]    = df["Close"].ewm(span=5,  adjust=False).mean()
    df["EMA20"]   = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]   = df["Close"].ewm(span=50, adjust=False).mean()
    df["LogRet"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["Ret1"]    = df["LogRet"]
    df["Ret5"]    = df["LogRet"].rolling(5).sum()
    df["Vol5"]    = df["LogRet"].rolling(5).std()
    df["Vol10"]   = df["LogRet"].rolling(10).std()
    df["Vol20"]   = df["LogRet"].rolling(20).std()
    df["CandlePos"] = (df["Close"] - df["Low"]) / ((df["High"] - df["Low"]) + 1e-9)
    df["BodyPct"]   = (df["Close"] - df["Open"]).abs() / ((df["High"] - df["Low"]) + 1e-9)
    df["MomentumZ"] = (
        (df["Ret5"] - df["Ret5"].rolling(20).mean()) /
        (df["Ret5"].rolling(20).std() + 1e-9)
    )

    if "^NSEI" in df.columns:
        idx = df["^NSEI"]
        df["IndexDD_60"] = (idx - idx.rolling(60).max()) / (idx.rolling(60).max() + 1e-9)
    else:
        df["IndexDD_60"] = 0.0

    df["DownVol_20"] = df["LogRet"].clip(upper=0).rolling(20).std()
    df["ATR"] = (
        df["High"] - df["Low"] +
        (df["High"] - df["Close"].shift(1)).abs() +
        (df["Low"]  - df["Close"].shift(1)).abs()
    ) / 2
    df["Gap%"]  = (df["Close"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9) * 100
    df["Gap"]   = (df["Close"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9)
    df["Range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["VolZ"]  = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-9)
    df["IntraRange"] = df["High"] - df["Low"]

    sec_idx_col = None
    for c in list(SECTOR_INDEX_INTERNAL.values()) + ["sector_index_value"]:
        if c in df.columns:
            numeric_check = pd.to_numeric(df[c], errors="coerce")
            if numeric_check.notna().sum() > 5 and numeric_check.std(skipna=True) > 0:
                sec_idx_col = c
                break
    if sec_idx_col:
        sec_ret = df[sec_idx_col].pct_change()
        df["RelativeRet5d"] = (df["Change %"].rolling(5).sum() - sec_ret.rolling(5).sum())
    else:
        df["RelativeRet5d"] = 0.0

    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    atr20 = df["ATR"].rolling(20).mean()
    df["KC_Mid"]   = ema20
    df["KC_Upper"] = ema20 + 2 * atr20
    df["KC_Lower"] = ema20 - 2 * atr20
    df["KC_Width"] = df["KC_Upper"] - df["KC_Lower"]
    df["KC_Pos"]   = (df["Close"] - df["KC_Lower"]) / (df["KC_Width"] + 1e-9)

    df["BB_Mid"]     = df["Close"].rolling(20).mean()
    df["BB_Std"]     = df["Close"].rolling(20).std()
    df["BB_Up"]      = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lo"]      = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Percent"] = (df["Close"] - df["BB_Lo"]) / (df["BB_Up"] - df["BB_Lo"] + 1e-9)

    df["NW20"] = nw_kernel_smooth(df["Close"])

    df["IPO_Recent_Flag"] = 0.0
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(dates) > 0:
            listing_date = dates.min()
            today = pd.Timestamp.today().normalize()
            if (today - listing_date).days < IPO_RECENT_DAYS:
                df["IPO_Recent_Flag"] = 1.0
    except Exception:
        pass

    df["High20"]    = df["Close"].rolling(20).max()
    df["Low20"]     = df["Close"].rolling(20).min()
    vol_ma          = df["Volume"].rolling(20).mean()
    df["VolRel"]    = df["Volume"] / (vol_ma + 1e-9)
    df["RevAfterHigh"] = (df["High20"] - df["Close"]) / (df["High20"] + 1e-9)
    df["RevAfterLow"]  = (df["Close"]  - df["Low20"]) / (df["Low20"]  + 1e-9)
    rng  = (df["High"] - df["Low"]).replace(0, np.nan)
    body = (df["Close"] - df["Open"]).abs()
    df["BreakReliab"] = (
        (body / (rng + 1e-9)).fillna(0) *
        (1 + df["VolRel"]) *
        (1 - df["RevAfterHigh"].clip(lower=0)) *
        (1 - df["RevAfterLow"].clip(lower=0))
    )

    df["VIX_Z"] = (
        (df["Vol10"] - df["Vol10"].rolling(20).mean()) /
        (df["Vol10"].rolling(20).std() + 1e-9)
    )
    
    df = add_raw_advanced_features(df, sec_idx_col=sec_idx_col)
    df = add_raw_statistics_and_calculus(df)
    df = add_family_ecosystem_features(df)
    df = add_indicator_internal_dots(df)
    df = add_price_volume_structure_dots(df)
    df = add_cross_family_dot_connections(df)
    df = add_final_market_evidence_scores(df)
    df = add_probability_ecosystem_features(df)

    for fc in fundamental_cols:
        if fc not in df.columns:
            df[fc] = 0.0
    df[fundamental_cols] = df[fundamental_cols].ffill().fillna(0.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill().dropna(subset=["Close"]).reset_index(drop=True)

    existing_base_feats = [
        "High", "Low", "Volume", "Change %", "RSI14", "EMA5", "EMA20", "EMA50",
        "Vol10", "ATR", "Gap%", "Gap", "Range", "VolZ", "IntraRange",
        "KC_Pos", "KC_Width", "BB_Percent", "NW20",
        "RevAfterHigh", "RevAfterLow", "BreakReliab",
        "RelativeRet5d", "VIX_Z", "Ret1", "Ret5", "Vol5", "Vol20",
        "CandlePos", "BodyPct", "MomentumZ", "IndexDD_60", "DownVol_20", "IPO_Recent_Flag"
    ]

    raw_advanced_feats = [
        # Trend / regime
        "ADX14", "PlusDI14", "MinusDI14", "DI_Spread",
        "Aroon_Up", "Aroon_Down", "Aroon_Osc",
        "Choppiness_Index",
        "Supertrend", "Supertrend_Direction",
        "HMA20", "KAMA20", "TRIX",

        # Momentum
        "MACD", "MACD_Signal", "MACD_Hist",
        "PPO", "ROC10", "ROC20",
        "TSI", "Stoch_K", "Stoch_D",
        "WilliamsR", "Ultimate_Oscillator",
        "Connors_RSI", "Fisher_Transform",

        # Volume / money flow
        "OBV", "OBV_Slope",
        "CMF20", "MFI14", "PVT",
        "Klinger", "Ease_Of_Movement",
        "VWAP", "VWAP_Distance",
        "Volume_Delta", "UpDown_Volume_Ratio",

        # Volatility / compression
        "ATR_Pct",
        "Historical_Volatility_20", "Historical_Volatility_60",
        "BB_Width", "BB_Squeeze",
        "Donchian_Width",
        "Parkinson_Volatility", "Garman_Klass_Volatility",
        "Ulcer_Index",

        # Market structure
        "Donchian_High20", "Donchian_Low20", "Donchian_Pos",
        "Distance_From_20D_High", "Distance_From_20D_Low",
        "Distance_From_52W_High", "Distance_From_52W_Low",
        "Breakout_Distance", "Pullback_Depth",

        # Relative strength
        "RS_Nifty_5D", "RS_Nifty_20D",
        "RS_Sector_5D", "RS_Sector_20D",
        "Rolling_Beta_60", "Rolling_Alpha_20",
        "Rolling_Correlation_Nifty_60", "Information_Ratio_60",
        "RS_Persistence_5", "RS_Persistence_20", "Alpha_Persistence_20",

        # Cross-timeframe
        "Weekly_RSI", "Weekly_MACD_Hist", "Weekly_EMA20_Slope",
        "Weekly_Trend_State", "Monthly_EMA20_Slope",
        "Daily_Inside_Weekly_Trend", "Daily_Weekly_Momentum_Alignment",

        # Liquidity
        "Dollar_Volume", "Amihud_Illiquidity",
        "Volume_Dryup_Ratio", "Liquidity_Shock", "Spread_Proxy",

        # Gap
        "Gap_vs_ATR", "Gap_Fill_Ratio",
        "Gap_Continuation_Flag", "Gap_Exhaustion_Score",
        "Opening_Gap_Strength",

        # Risk / drawdown
        "Downside_Semivariance_20",
        "Max_Drawdown_20", "Max_Drawdown_60",
        "Drawdown_Speed", "Crash_Risk_Score",
        "Left_Tail_Return_Count_20",

        # Exhaustion
        "Trend_Age", "Consecutive_Up_Days", "Consecutive_Down_Days",
        "Distance_From_EMA20_ATR", "Distance_From_EMA50_ATR",
        "Overextension_Score",

        # Compression / expansion
        "Range_Compression_5_20",
        "Volume_Compression_5_20",
        "Volatility_Compression_20",
        "Squeeze_Intensity",
        "Expansion_Breakout_Score",

        # Candle sequence
        "Bullish_Candle_Streak", "Bearish_Candle_Streak",
        "Higher_Close_Count_5", "Lower_Close_Count_5",
        "Inside_Bar_Count_10",
        "Wide_Range_Bar_Flag", "Narrow_Range_Bar_Flag"
    ]

    raw_statistical_feats = [
        "Robust_Return_Z20",
        "Rolling_Median_Return_20",
        "Rolling_MAD_Return_20",
        "Return_IQR_60",
        "Rolling_Skew_20",
        "Rolling_Kurtosis_20",
        "Rolling_Sharpe_20",
        "Rolling_TStat_Return_20",
        "Entropy_Return_20",
        "Autocorr_Return_1_20",
        "Autocorr_Return_5_60",
        "Hurst_Exponent_60",
        "Variance_Ratio_20",
        "RSI_Percentile_60",
        "ATR_Percentile_60",
        "Volume_Percentile_60",
        "Range_Percentile_60",
        "Volatility_Percentile_60",
        "Beta_Stability_60",
        "Rolling_Cov_Stock_Nifty_60"
    ]

    raw_calculus_feats = [
        "Price_Slope_5",
        "Price_Slope_20",
        "EMA20_Slope",
        "EMA50_Slope",
        "EMA20_Acceleration",
        "Rolling_Linear_Trend_R2_20",
        "Rolling_Quadratic_Curvature_20",
        "Trend_Convexity_20",
        "Price_Inflection_Flag",
        "RSI_Velocity",
        "RSI_Acceleration",
        "MACD_Hist_Velocity",
        "MACD_Hist_Acceleration",
        "RSI_Turning_Point",
        "MACD_Hist_Turning_Point",
        "OBV_Velocity",
        "CMF_Velocity",
        "ATR_Velocity",
        "Volatility_Slope_20",
        "Volatility_Acceleration_20",
        "Volume_Acceleration",
        "Drawdown_Velocity",
        "Drawdown_Acceleration"
    ]

    family_names = [
        "Trend", "Momentum", "VolumeFlow", "Volatility", "Structure",
        "RelativeStrength", "CrossTimeframe", "Liquidity", "Gap",
        "RiskDrawdown", "Exhaustion", "Compression", "CandleSequence"
    ]

    family_score_feats = [f"{fam}_Score" for fam in family_names]

    family_stat_calc_feats = []
    for fam in family_names:
        family_stat_calc_feats += [
            f"{fam}_Z20",
            f"{fam}_Z60",
            f"{fam}_Percentile_60",
            f"{fam}_RollingMean_20",
            f"{fam}_RollingStd_20",
            f"{fam}_Autocorr_20",
            f"{fam}_Persistence_20",
            f"{fam}_Signal_Stability",
            f"{fam}_Noise_Ratio",
            f"{fam}_Price_Correlation_20",
            f"{fam}_Price_Correlation_60",
            f"{fam}_Lead_Return_Correlation_20",
            f"{fam}_Velocity",
            f"{fam}_Acceleration",
            f"{fam}_Curvature",
            f"{fam}_Turning_Point",
            f"{fam}_Slope_5",
            f"{fam}_Slope_20",
            f"{fam}_Divergence_From_Price"
        ]

    ecosystem_relation_feats = [
        "Trend_Momentum_Agreement",
        "Trend_Volume_Confirmation",
        "Trend_Volatility_Compatibility",
        "Trend_Structure_Alignment",
        "Trend_RelativeStrength_Alignment",
        "Trend_CrossTimeframe_Alignment",

        "Momentum_Volume_Confirmation",
        "Momentum_Volatility_Compatibility",
        "Momentum_Structure_Alignment",
        "Momentum_RelativeStrength_Alignment",
        "Momentum_CrossTimeframe_Alignment",

        "Volume_Volatility_Pressure",
        "Volume_Structure_Confirmation",
        "Volume_RelativeStrength_Confirmation",
        "Volume_Liquidity_Quality",

        "Volatility_Structure_Pressure",
        "Volatility_Compression_Pressure",
        "Volatility_Risk_Alignment",
        "Volatility_Exhaustion_Risk",

        "Structure_RelativeStrength_Alignment",
        "Structure_CrossTimeframe_Alignment",
        "Structure_Compression_Breakout_Readiness",
        "Structure_Gap_Compatibility",

        "Risk_Trend_Conflict",
        "Risk_Momentum_Conflict",
        "Exhaustion_Trend_Conflict",
        "Exhaustion_Momentum_Conflict",

        "Ecosystem_Agreement_Index",
        "Ecosystem_Conflict_Index",
        "Ecosystem_Directional_Bias",
        "Ecosystem_Breakout_Readiness",
        "Ecosystem_Reversal_Risk",
        "Ecosystem_Noise_Level"
    ]

    ecosystem_probability_feats = [
        "Ecosystem_State_ID",
        "Ecosystem_State_Frequency",
        "Ecosystem_State_Rarity",

        "P_Up_Given_Ecosystem_State",
        "P_Down_Given_Ecosystem_State",
        "P_Flat_Given_Ecosystem_State",

        "Expected_Return_Given_Ecosystem_State",
        "Expected_AbsReturn_Given_Ecosystem_State",
        "Expected_Downside_Given_Ecosystem_State",
        "Expected_Upside_Given_Ecosystem_State",

        "WinLoss_Odds_Given_Ecosystem_State",
        "Payoff_Ratio_Given_Ecosystem_State",
        "Ecosystem_Edge",
        "Ecosystem_Uncertainty",

        "Prob_State_Continuation",
        "Prob_State_Reversal",
        "Prob_Breakout_Transition",
        "Prob_Breakdown_Transition",
        "Prob_MeanReversion_Transition",
        "Prob_Volatility_Expansion",
        "Prob_Volatility_Compression",

        "State_Sample_Confidence",
        "State_Probability_Stability",
        "State_Edge_Stability"
    ]
    internal_dot_feats = [
        # RSI dots
        "RSI_Above_50_State", "RSI_50_Cross_Up", "RSI_50_Cross_Down",
        "RSI_Above_55_State", "RSI_Below_45_State", "RSI_Neutral_45_55",
        "RSI_Escape_Above_55", "RSI_Escape_Below_45",
        "RSI_MA14", "RSI_MA_Spread", "RSI_MA_Cross_Up", "RSI_MA_Cross_Down",
        "RSI_Overbought_70", "RSI_Oversold_30", "RSI_Exit_Overbought", "RSI_Exit_Oversold",
        "RSI_Velocity_Cross_Up", "RSI_Velocity_Cross_Down",
        "RSI_Acceleration_Positive", "RSI_Acceleration_Negative",
        "RSI_Bullish_Evidence", "RSI_Bearish_Evidence",
        "RSI_Exhaustion_Evidence", "RSI_Reversal_Evidence",

        # EMA / trend dots
        "Price_Above_EMA20", "Price_Above_EMA50", "EMA20_Above_EMA50",
        "EMA20_EMA50_Cross_Up", "EMA20_EMA50_Cross_Down",
        "EMA20_Slope_Positive", "EMA50_Slope_Positive",
        "EMA20_Acceleration_Positive", "EMA20_Acceleration_Negative",
        "EMA20_Distance_ATR", "EMA50_Distance_ATR",
        "EMA20_Overextended_Up", "EMA20_Overextended_Down",
        "Trend_Age_Bullish", "Trend_Age_Bearish",
        "Fresh_Bullish_Trend", "Fresh_Bearish_Trend",
        "Mature_Bullish_Trend", "Mature_Bearish_Trend",
        "EMA_Bullish_Trend_Evidence", "EMA_Bearish_Trend_Evidence",
        "EMA_Trend_Strength_Evidence", "EMA_Overextension_Evidence",

       # MACD dots
        "MACD_Above_Signal", "MACD_Below_Signal",
        "MACD_Cross_Up", "MACD_Cross_Down",
        "MACD_Above_Zero", "MACD_Below_Zero",
        "MACD_Hist_Positive", "MACD_Hist_Negative",
        "MACD_Hist_Rising", "MACD_Hist_Falling",
        "MACD_Hist_Zero_Cross_Up", "MACD_Hist_Zero_Cross_Down",
        "MACD_Bullish_Evidence", "MACD_Bearish_Evidence",
        "MACD_Momentum_Acceleration_Evidence", "MACD_Momentum_Deceleration_Evidence",

        # Candle dots
        "Candle_Body_Pct", "Candle_Upper_Wick_Pct", "Candle_Lower_Wick_Pct",
        "Candle_Close_Position", "Bullish_Candle", "Bearish_Candle",
        "Strong_Bullish_Candle", "Strong_Bearish_Candle", "Indecision_Candle",
        "Upper_Wick_Rejection", "Lower_Wick_Rejection",
        "Buyer_Control_Candle", "Seller_Control_Candle",
        "Wide_Range_Candle", "Narrow_Range_Candle",
        "Inside_Bar_Dot", "Outside_Bar_Dot",
        "Bullish_Candle_Streak_Dot", "Bearish_Candle_Streak_Dot",
        "Candle_Buyer_Control_Evidence", "Candle_Seller_Control_Evidence",
        "Candle_Rejection_Evidence", "Candle_Indecision_Evidence",
        "Candle_Breakout_Evidence",

        # Volume dots
        "Volume_Above_20D_Avg", "Volume_Expansion_Dot", "Volume_Dryup_Dot",
        "Volume_Percentile_High", "Volume_Percentile_Low",
        "OBV_Slope_Positive", "OBV_Slope_Negative",
        "CMF_Positive_Dot", "CMF_Negative_Dot",
        "MFI_Above_50", "MFI_Below_50",
        "Volume_Delta_Positive_Dot", "Volume_Delta_Negative_Dot",
        "UpDown_Volume_Ratio_Strong", "UpDown_Volume_Ratio_Weak",
        "Volume_Bullish_Evidence", "Volume_Bearish_Evidence",
        "Volume_Confirmation_Evidence", "Volume_Divergence_Evidence",
        "Liquidity_Shock_Evidence",

        # Volatility / compression dots
        "ATR_Pct_High", "ATR_Pct_Low",
        "Volatility_Compression_Dot", "Volatility_Expansion_Dot",
        "BB_Squeeze_Active", "BB_Expansion_Active",
        "Range_Compression_Dot", "Range_Expansion_Dot",
        "Compression_Building", "Compression_Release",
        "Squeeze_Intensity_High",
        "Volatility_Expansion_Evidence", "Compression_Breakout_Evidence",
        "High_Risk_Volatility_Evidence", "Low_Noise_Compression_Evidence",

        # Structure dots
        "Near_20D_High", "Near_20D_Low", "Near_52W_High", "Near_52W_Low",
        "Donchian_Upper_Break", "Donchian_Lower_Break",
        "Donchian_Mid_Above", "Donchian_Mid_Below",
        "Breakout_Distance_Positive", "Breakdown_Distance_Negative",
        "Pullback_Depth_High",
        "Structure_Bullish_Evidence", "Structure_Bearish_Evidence",
        "Breakout_Evidence", "Breakdown_Evidence",
        "Pullback_Evidence", "Support_Reaction_Evidence", "Resistance_Rejection_Evidence",

        # Relative strength dots
        "RS_Nifty_Positive", "RS_Nifty_Negative",
        "RS_Sector_Positive", "RS_Sector_Negative",
        "RS_Nifty_Improving", "RS_Nifty_Weakening",
        "RS_Sector_Improving", "RS_Sector_Weakening",
        "Alpha_Positive_Dot", "Alpha_Negative_Dot", "Alpha_Persistence_Strong",
        "RelativeStrength_Bullish_Evidence", "RelativeStrength_Bearish_Evidence",
        "Leadership_Evidence", "Weakness_Evidence",

        # Gap dots
        "Gap_Up_Dot", "Gap_Down_Dot", "Large_Gap_Up", "Large_Gap_Down",
        "Gap_Filled_Dot", "Gap_Not_Filled_Dot",
        "Gap_Continuation_Bullish", "Gap_Continuation_Bearish",
        "Gap_Exhaustion_Dot",
        "Gap_Bullish_Evidence", "Gap_Bearish_Evidence", "Gap_Rejection_Evidence",

        # Risk / exhaustion dots
        "Overextension_High", "Distance_From_EMA20_ATR_High", "Distance_From_EMA50_ATR_High",
        "Consecutive_Up_Days_High", "Consecutive_Down_Days_High",
        "Crash_Risk_High", "Drawdown_Speed_High", "Left_Tail_Risk_High",
        "Exhaustion_Risk_High", "Risk_Bearish_Evidence",
        "Risk_Reversal_Evidence", "Trend_Maturity_Evidence", "No_Trade_Risk_Evidence"
    ]
    
    cross_family_dot_feats = [
        "RSI_Trend_Bullish_Agreement", "RSI_Trend_Bearish_Agreement",
        "RSI_Trend_Conflict", "RSI_Overbought_Trend_Strength",
        "RSI_Overbought_Exhaustion_Risk", "RSI_Oversold_Reversal_Setup",
    
        "RSI_Candle_Bullish_Confirmation", "RSI_Candle_Bearish_Confirmation",
        "RSI_Candle_Reversal_Evidence",
        "RSI_Exit_Oversold_Bullish_Candle", "RSI_Exit_Overbought_Bearish_Candle",

        "RSI_Volume_Bullish_Confirmation", "RSI_Volume_Bearish_Confirmation",
        "RSI_Activation_With_Volume", "RSI_Activation_Without_Volume_Risk",

        "Trend_Volume_Bullish_Confirmation", "Trend_Volume_Bearish_Confirmation",
        "Trend_Without_Volume_Risk", "Volume_Against_Trend_Warning",

        "Trend_Candle_Bullish_Confirmation", "Trend_Candle_Bearish_Confirmation",
        "Trend_Candle_Rejection_Warning", "Trend_Candle_Pullback_Opportunity",

        "Breakout_Volume_Confirmation", "Breakout_Without_Volume_Risk",
        "Breakout_Failure_Risk", "False_Breakout_Evidence",

        "Breakdown_Volume_Confirmation", "Breakdown_Without_Volume_Risk",
    
        "Compression_Breakout_Readiness", "Squeeze_Release_Evidence",
        "Volatility_Expansion_Setup", "Post_Compression_Failure_Risk",

        "Trend_RS_Leadership", "Trend_RS_Weakness",
        "Bullish_Trend_With_Sector_Leadership", "Bearish_Trend_With_Sector_Weakness",

        "Trend_Exhaustion_Risk", "Trend_Continuation_Quality",
        "Overextension_Reversal_Risk", "Healthy_Trend_Pullback",

        "Gap_Candle_Continuation_Confirmation", "Gap_Candle_Rejection_Warning",
        "Gap_Volume_Continuation_Evidence", "Gap_Exhaustion_With_Weak_Close",

        "Momentum_Stack_Bullish", "Momentum_Stack_Bearish", "Momentum_Trend_Disagreement"
    ]

    final_evidence_feats = [
        "Bullish_Evidence_Total",
        "Bearish_Evidence_Total",
        "Net_Directional_Evidence",
        "Trend_Continuation_Evidence",
        "Reversal_Evidence",
        "Breakout_Readiness_Evidence",
        "Breakdown_Readiness_Evidence",
        "False_Breakout_Risk",
        "Volatility_Expansion_Evidence_Final",
        "Exhaustion_Risk_Evidence",
        "Trade_Quality_Evidence",
        "No_Trade_Noise_Evidence"
    ]    
    base_feats = (
        existing_base_feats
        + raw_advanced_feats
        + raw_statistical_feats
        + raw_calculus_feats
        + family_score_feats
        + family_stat_calc_feats
        + ecosystem_relation_feats
        + internal_dot_feats
        + cross_family_dot_feats
        + final_evidence_feats
        + ecosystem_probability_feats
    )
    fundamental_used = [f for f in fundamental_cols if f in df.columns]

    market_idx_feats = []
    for c in ["^NSEI", "^BSESN"] + list(SECTOR_INDEX_INTERNAL.values()):
        if c in df.columns:
            market_idx_feats.append(c)

    feature_cols = base_feats + fundamental_used + list(dict.fromkeys(market_idx_feats))
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]
    log.info(f"[Feature Engineering] Total features generated: {len(feature_cols)}")
    break_cols = ["BreakReliab", "RevAfterHigh", "RevAfterLow"]
    return df, feature_cols, break_cols


# ---------------------------------------------------------------
# Load a single CSV
# ---------------------------------------------------------------
def load_csv_by_symbol(path):
    df_raw = pd.read_csv(path, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    if "symbol" in df_raw.columns:
        syms   = df_raw["symbol"].unique()
        result = {}
        for sym in syms:
            sym_df = df_raw[df_raw["symbol"] == sym].copy().reset_index(drop=True)
            result[str(sym).upper()] = sym_df
        return result
    else:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        stem = re.sub(r'(?i)_with_nifty.*$', '', stem)
        stem = re.sub(r'(?i)[._]NS$',  '', stem)
        stem = re.sub(r'(?i)[._]BSE$', '', stem)
        stem = re.sub(r'(?i)[._]BO$',  '', stem)
        sym  = re.sub(r'[^A-Za-z0-9]', '', stem).upper()
        return {sym: df_raw}

def merge_small_buckets(buckets, min_size=20):
    merged       = {}
    generic_pool = []
    for (vol_level, sector), items in buckets.items():
        if len(items) < min_size:
            generic_pool.extend(items)
        else:
            merged[(vol_level, sector)] = items
    if generic_pool:
        vol_groups = {}
        for item in generic_pool:
            sym, df, fcols, bcols = item
            annual_vol = calculate_annualized_volatility(df)
            vl         = get_volatility_level(annual_vol)
            vol_groups.setdefault(vl, []).append(item)
        for vl, items_list in vol_groups.items():
            key = (vl, "GENERIC")
            if key in merged:
                merged[key].extend(items_list)
            else:
                merged[key] = items_list
    return merged

# ---------------------------------------------------------------
# Beta / momentum helpers
# ---------------------------------------------------------------
def compute_beta(df, nifty_col="^NSEI", window=252):
    if nifty_col not in df.columns:
        return np.nan
    stock_ret  = df["Close"].pct_change().dropna()
    market_ret = df[nifty_col].pct_change().dropna()
    common_idx = stock_ret.index.intersection(market_ret.index)
    if len(common_idx) < 60:
        return np.nan
    sr  = stock_ret.loc[common_idx]
    mr  = market_ret.loc[common_idx]
    cov = np.cov(sr, mr)[0, 1]
    var = np.var(mr) + 1e-12
    return cov / var

def compute_momentum_score(df, lookback=126):
    if len(df) < lookback + 1:
        return 0.0
    return float(df["Close"].iloc[-1] / df["Close"].iloc[-(lookback+1)] - 1)

def compute_mean_reversion_score(df, lookback=20):
    if len(df) < lookback + 1:
        return 0.0
    ma = df["Close"].rolling(lookback).mean().iloc[-1]
    sd = df["Close"].rolling(lookback).std().iloc[-1] + 1e-9
    return float((df["Close"].iloc[-1] - ma) / sd)

HIGH_BETA_THRESH      = 1.3
LOW_BETA_THRESH       = 0.7
HIGH_MOMENTUM_THRESH  = 0.15
MEAN_REV_ZSCORE       = -1.5

# ---------------------------------------------------------------
# Assign bucket
# ---------------------------------------------------------------
def assign_bucket(sym, df_raw):
    df_tmp = df_raw.copy()
    if "date" in df_tmp.columns:
        df_tmp.rename(columns={"date": "Date"}, inplace=True)
    df_feat, _, _ = build_features_from_df(df_tmp)
    annual_vol    = calculate_annualized_volatility(df_feat)
    vol_level     = get_volatility_level(annual_vol)
    last          = df_tmp.iloc[-1]
    sec_str = str(last.get("sector",   "")) if "sector"   in df_tmp.columns else ""
    ind_str = str(last.get("industry", "")) if "industry" in df_tmp.columns else ""
    sector  = map_sector_from_metadata(sec_str, ind_str)
    return (vol_level, sector)


# ---------------------------------------------------------------
# Direction labels (1-day)
# ---------------------------------------------------------------
def make_dir_labels_1d(y1_fwd, close, high, low, atr_period=14):
    close      = close.astype(float)
    high       = high.astype(float)
    low        = low.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr     = tr.rolling(atr_period).mean().bfill()
    atr_pct = (atr / (close + 1e-9)).values.reshape(-1, 1)
    atr_pct = np.clip(atr_pct, 1e-6, None)
    norm_ret = y1_fwd / atr_pct
    thresh   = 0.25
    dir_all  = np.full_like(norm_ret, 0.5, dtype=float)
    dir_all[norm_ret >  thresh] = 1.0
    dir_all[norm_ret < -thresh] = 0.0
    return dir_all


# ---------------------------------------------------------------
# Sequence builder (disk-based, for pretrain)
# Cache keys include symbol + stage + cutoff_date + seq_len so
# yearly / monthly / weekly / daily caches never collide.
# ---------------------------------------------------------------
def build_and_save_sequences_for_stock(df, feature_cols_union, symbol, npz_dir,
                                       seq_len=SEQ_LEN, stage="yearly_pretrain",
                                       cutoff_date=None):
    for c in feature_cols_union:
        if c not in df.columns:
            df[c] = 0.0

    X_all   = df[feature_cols_union].values.astype(np.float32)
    X_all   = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    y1_raw  = df[["LogRet"]].values.astype(np.float32)
    y1_fwd  = cumulative_logret_forward(y1_raw, horizon=1).astype(np.float32)
    dir_all = make_dir_labels_1d(y1_fwd, df["Close"], df["High"], df["Low"])

    mask_valid = ~np.isnan(y1_fwd.ravel())
    if not mask_valid.any() or mask_valid.sum() < 20:
        log.info(f"  [Skip] {symbol}: insufficient valid samples ({mask_valid.sum()})")
        return None

    last_valid = np.where(mask_valid)[0].max()
    X_all      = X_all[:last_valid+1]
    y1_all     = y1_fwd[:last_valid+1]
    dir_all    = dir_all[:last_valid+1]
    mask_valid = mask_valid[:last_valid+1]
    B_all = df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].values.astype(np.float32)[:last_valid+1]

    n       = len(X_all)
    min_val = max(20, min(60, n // 5))
    cut     = n - min_val
    cut_lo  = max(0, cut - EMBARGO_STEPS)
    cut_hi  = min(n, cut + EMBARGO_STEPS)

    m_tr  = np.zeros(n, dtype=bool); m_tr[:cut_lo]  = True; m_tr  &= mask_valid
    m_val = np.zeros(n, dtype=bool); m_val[cut_hi:] = True; m_val &= mask_valid

    if not m_tr.any():
        log.info(f"  [Skip seq] {symbol}: no train samples")
        return None

    x_scaler = MinMaxScaler().fit(X_all[m_tr])
    b_scaler = MinMaxScaler().fit(B_all[m_tr])
    X_scaled = x_scaler.transform(X_all).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    B_scaled = b_scaler.transform(B_all).astype(np.float32)
    B_scaled = np.nan_to_num(B_scaled, nan=0.0)

    X_scaled = np.clip(X_scaled, -500.0, 500.0).astype(np.float32)
    B_scaled = np.clip(B_scaled, -500.0, 500.0).astype(np.float32)

    pack_tr, _  = make_sequences_masked(X_scaled, {"y1": y1_all, "dir": dir_all.reshape(-1,1), "B": B_scaled}, seq_len, m_tr)
    pack_val, _ = make_sequences_masked(X_scaled, {"y1": y1_all, "dir": dir_all.reshape(-1,1), "B": B_scaled}, seq_len, m_val)

    if "X" not in pack_tr or pack_tr["X"].shape[0] < 5:
        log.info(f"  [Skip seq] {symbol}: too few sequences ({pack_tr['X'].shape[0]})")
        return None

    y1_mu   = pack_tr["y1"].mean(); y1_sd = pack_tr["y1"].std() + 1e-9
    int_tr  = np.tanh((pack_tr["y1"]  - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    int_val = (np.tanh((pack_val["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
               if pack_val["X"].shape[0] > 0 else np.zeros((0,1), dtype=np.float32))

    dir_tr  = apply_label_smoothing(pack_tr["dir"]).astype(np.float32)
    dir_val = (apply_label_smoothing(pack_val["dir"]).astype(np.float32)
               if pack_val["X"].shape[0] > 0 else np.zeros((0,1), dtype=np.float32))

    # Cache key includes stage and cutoff_date to prevent cross-stage collisions
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date) if cutoff_date else "nodate"
    cache_tag  = f"{symbol}_{stage}_{cutoff_str}_sl{seq_len}"

    os.makedirs(npz_dir, exist_ok=True)
    train_fp = os.path.join(npz_dir, f"{cache_tag}_train.npz")
    np.savez_compressed(train_fp,
        X=pack_tr["X"].astype(np.float32), B=pack_tr["B"].astype(np.float32),
        y1=pack_tr["y1"].astype(np.float32), dir=dir_tr, inten=int_tr)

    val_fp = None
    if pack_val["X"].shape[0] > 0:
        val_fp = os.path.join(npz_dir, f"{cache_tag}_val.npz")
        np.savez_compressed(val_fp,
            X=pack_val["X"].astype(np.float32), B=pack_val["B"].astype(np.float32),
            y1=pack_val["y1"].astype(np.float32), dir=dir_val, inten=int_val)

    return train_fp, val_fp


# ================================================================
# MODEL LAYERS
# ================================================================

class GatingLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units     = units
        self.gate      = layers.Dense(units, activation='sigmoid')
        self.transform = layers.Dense(units, activation='elu')
        self._proj     = None

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != self.units:
            self._proj = layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        gate      = self.gate(inputs)
        transform = self.transform(inputs)
        residual  = self._proj(inputs) if self._proj is not None else inputs
        return gate * transform + (1 - gate) * residual


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def call(self, inputs):
        seq_len   = tf.shape(inputs)[1]
        positions = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
        angles    = tf.cast(tf.range(self.d_model), tf.float32)[tf.newaxis, :]
        angle_rates  = 1 / tf.pow(10000.0, (2 * (angles // 2)) / tf.cast(self.d_model, tf.float32))
        pos_encoding = positions * angle_rates
        even_mask    = tf.range(self.d_model) % 2 == 0
        pos_encoding = tf.where(even_mask, tf.sin(pos_encoding), tf.cos(pos_encoding))
        pos_encoding = tf.cast(pos_encoding, inputs.dtype)
        return inputs + pos_encoding[tf.newaxis, :seq_len, :]


# ================================================================
# ENCODER
# ================================================================

def build_advanced_encoder(seq_len, n_features, reg=L2REG, dropout_rate=DROPOUT_ENC_PRETRAIN):
    inputs = layers.Input(shape=(seq_len, n_features), name="enc_input")
    x_in   = tf.cast(inputs, tf.float16)

    cnn3 = layers.Conv1D(96, 3, padding="same", activation="relu",
                         kernel_regularizer=regularizers.l2(reg))(x_in)
    cnn3 = layers.BatchNormalization()(cnn3)
    cnn3 = layers.Dropout(dropout_rate)(cnn3)

    cnn5 = layers.Conv1D(96, 5, padding="same", activation="relu",
                         kernel_regularizer=regularizers.l2(reg))(x_in)
    cnn5 = layers.BatchNormalization()(cnn5)
    cnn5 = layers.Dropout(dropout_rate)(cnn5)

    cnn7 = layers.Conv1D(96, 7, padding="same", activation="relu",
                         kernel_regularizer=regularizers.l2(reg))(x_in)
    cnn7 = layers.BatchNormalization()(cnn7)
    cnn7 = layers.Dropout(dropout_rate)(cnn7)

    cnn_concat = layers.Concatenate(axis=-1)([cnn3, cnn5, cnn7])

    d_model = 192
    x = layers.Dense(d_model, activation="relu",
                     kernel_regularizer=regularizers.l2(reg))(cnn_concat)
    x = layers.LayerNormalization(epsilon=1e-3)(x)
    x = PositionalEncoding(seq_len, d_model)(x)

    x_expanded = tf.expand_dims(x, axis=2)
    x_expanded = tf.tile(x_expanded, [1, 1, n_features, 1])
    var_selector = layers.Dense(1, activation='sigmoid')(x_expanded)
    var_selector = tf.squeeze(var_selector, axis=-1)
    var_weights  = layers.Softmax(axis=-1)(var_selector)
    var_weights  = tf.expand_dims(var_weights, axis=-1)
    x = tf.reduce_sum(x_expanded * var_weights, axis=2)

    for _ in range(4):
        attn_output = layers.MultiHeadAttention(
            num_heads=6, key_dim=d_model // 8,
            dropout=dropout_rate
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-3)(x + attn_output)
        ffn = layers.Dense(d_model * 2, activation="relu",
                           kernel_regularizer=regularizers.l2(reg))(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(d_model, kernel_regularizer=regularizers.l2(reg))(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        gate = layers.Dense(d_model, activation='sigmoid')(x)
        x    = layers.LayerNormalization(epsilon=1e-3)(x + gate * ffn)

    temporal_scores  = layers.Dense(1, activation='sigmoid')(x)
    temporal_weights = layers.Softmax(axis=1)(temporal_scores)
    weighted_context = tf.reduce_sum(x * temporal_weights, axis=1)

    lstm_out = layers.LSTM(96, return_sequences=False,
                           kernel_regularizer=regularizers.l2(reg))(x)
    lstm_out = layers.Dropout(dropout_rate)(lstm_out)

    combined = layers.Concatenate()([weighted_context, lstm_out])
    output   = layers.Dense(96, activation="relu",
                            kernel_regularizer=regularizers.l2(reg))(combined)
    output   = layers.Dropout(DROPOUT_HEAD)(output)

    return tf.keras.Model(inputs, output, name="advanced_encoder")


def add_adapter(x, r=16):
    d    = x.shape[-1]
    down = layers.Dense(max(4, int(d) // r), activation="relu")(x)
    up   = layers.Dense(int(d))(down)
    return layers.Add()([x, up])


def _build_head(seq_len, n_features, dropout_enc, model_name):
    enc = build_advanced_encoder(seq_len, n_features, reg=L2REG, dropout_rate=dropout_enc)

    inp_price = layers.Input(shape=(seq_len, n_features), name="price_seq")
    z = enc(inp_price)
    z = add_adapter(z)

    inp_break  = layers.Input(shape=(3,), name="break_feats")
    break_cast = tf.cast(inp_break, tf.float16)

    inp_vol_level = layers.Input(shape=(1,), name="vol_level")
    vol_emb = layers.Embedding(
        input_dim=len(VOLATILITY_LEVELS) + 2, output_dim=8
    )(inp_vol_level)
    vol_emb = layers.Flatten()(vol_emb)
    vol_emb = tf.cast(vol_emb, tf.float16)

    m = layers.Concatenate()([z, break_cast, vol_emb])
    m = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2REG))(m)
    m = layers.Dropout(DROPOUT_HEAD)(m)
    m = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(L2REG))(m)
    m = layers.Dropout(DROPOUT_HEAD)(m)

    m_fp32  = tf.cast(m, tf.float32)
    out_r1  = layers.Dense(1, dtype="float32", name="r1")(m_fp32)
    out_dir = layers.Dense(1, activation="sigmoid", dtype="float32", name="dir")(m_fp32)
    out_int = layers.Dense(1, activation="tanh",    dtype="float32", name="int")(m_fp32)

    return tf.keras.Model([inp_price, inp_break, inp_vol_level],
                          [out_r1, out_dir, out_int], name=model_name)


def build_pretrain_model(seq_len, n_features, dropout_enc=DROPOUT_ENC_PRETRAIN):
    return _build_head(seq_len, n_features, dropout_enc, "pretrain_model")

def build_multitask_model(seq_len, n_features, dropout_enc=DROPOUT_ENC_FINETUNE):
    return _build_head(seq_len, n_features, dropout_enc, "finetune_model")


# ---------------------------------------------------------------
# Losses & weights
# ---------------------------------------------------------------
def spike_weighted_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w = 1.5 + 3.5 * tf.pow(tf.abs(y_true), 1.5)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))

def smooth_huber(y_true, y_pred, delta=0.05):
    y_true  = tf.cast(y_true, tf.float32)
    y_pred  = tf.cast(y_pred, tf.float32)
    err     = y_true - y_pred
    abs_err = tf.abs(err)
    loss    = tf.where(abs_err < delta, 0.5 * tf.square(err) / delta, abs_err - 0.5 * delta)
    return tf.reduce_mean(loss)

def focal_bce_soft(y_true, y_pred, alpha=0.5, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    eps    = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    w      = alpha * tf.pow(1.0 - pt, gamma)
    return -tf.reduce_mean(w * tf.math.log(pt))

LOSSES       = {"r1": spike_weighted_mse, "dir": focal_bce_soft, "int": smooth_huber}
LOSS_W_PRE   = {"r1": 0.20, "dir": 0.70, "int": 0.10}
LOSS_W_FT_M  = {"r1": 0.22, "dir": 0.65, "int": 0.13}   # monthly
LOSS_W_FT    = {"r1": 0.25, "dir": 0.60, "int": 0.15}   # weekly
LOSS_W_DAILY = {"r1": 0.30, "dir": 0.55, "int": 0.15}   # daily


# ---------------------------------------------------------------
# Pretrain per bucket (streaming)
# ---------------------------------------------------------------
def pretrain_bucket(items, feature_cols_union, vol_level, sector, out_dir,
                    cutoff_date=None):
    os.makedirs(out_dir, exist_ok=True)
    npz_dir = os.path.join(out_dir, "npz_cache")
    os.makedirs(npz_dir, exist_ok=True)

    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}
    vol_idx         = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

    train_files, val_files = [], []
    for (sym, df, _fcols, _bcols) in items:
        res = build_and_save_sequences_for_stock(
            df, feature_cols_union, sym, npz_dir,
            seq_len=SEQ_LEN, stage="yearly_pretrain", cutoff_date=cutoff_date
        )
        if res is None:
            continue
        tr_fp, val_fp = res
        if tr_fp:  train_files.append(tr_fp)
        if val_fp: val_files.append(val_fp)

    if not train_files:
        log.info(f"  [Skip] No train sequences for bucket {vol_level}/{sector}")
        return None

    n_features = len(feature_cols_union)

    def make_ds(file_list, shuffle=False):
        def gen():
            for fp in file_list:
                data = np.load(fp)
                X, B = data["X"].astype(np.float32), data["B"].astype(np.float32)
                y1, d, inten = (data["y1"].astype(np.float32),
                                data["dir"].astype(np.float32),
                                data["inten"].astype(np.float32))
                for i in range(X.shape[0]):
                    yield (
                        {"price_seq":   X[i],
                         "break_feats": B[i],
                         "vol_level":   np.array([vol_idx], dtype=np.int32)},
                        {"r1": y1[i].reshape(1,), "dir": d[i].reshape(1,), "int": inten[i].reshape(1,)}
                    )
        ds = tf.data.Dataset.from_generator(gen, output_signature=(
            {"price_seq":   tf.TensorSpec((SEQ_LEN, n_features), tf.float32),
             "break_feats": tf.TensorSpec((3,), tf.float32),
             "vol_level":   tf.TensorSpec((1,), tf.int32)},
            {"r1":  tf.TensorSpec((1,), tf.float32),
             "dir": tf.TensorSpec((1,), tf.float32),
             "int": tf.TensorSpec((1,), tf.float32)}
        ))
        if shuffle:
            ds = ds.shuffle(20000)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_files, shuffle=True)
    val_ds   = make_ds(val_files,   shuffle=False) if val_files else None

    model     = build_pretrain_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_PRETRAIN)
    ckpt_path = os.path.join(out_dir, "pretrain_best.weights.h5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PT, clipnorm=1.0),
        loss=LOSSES, loss_weights=LOSS_W_PRE
    )

    callbacks = [
        EarlyStopping(monitor="val_loss" if val_ds else "loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss" if val_ds else "loss",
                        save_best_only=True, save_weights_only=True, verbose=0)
    ]

    model.fit(train_ds, epochs=EPOCHS_PT, validation_data=val_ds, callbacks=callbacks, verbose=1)

    enc      = model.get_layer("advanced_encoder")
    enc_path = os.path.join(out_dir, "encoder.weights.h5")
    enc.save_weights(enc_path)

    bucket_tag = f"{vol_level}_{sector}"
    window_start = df["Date"].min() if "Date" in items[0][1].columns else None
    window_end   = cutoff_date
    save_stage_metadata(out_dir, "yearly_pretrain", cutoff_date,
                        window_start, window_end, None, bucket_tag)
    return enc_path


# ---------------------------------------------------------------
# Prepare arrays for single stock (shared by monthly/weekly/daily)
# ---------------------------------------------------------------
def prepare_single_stock_arrays(df, feature_cols, seq_len):
    X_all   = df[feature_cols].values.astype(float)
    y1_raw  = df[["LogRet"]].values.astype(float)
    y1_fwd  = cumulative_logret_forward(y1_raw, horizon=1)
    y1_all  = y1_fwd
    dir_all = make_dir_labels_1d(y1_fwd, df["Close"], df["High"], df["Low"])

    mask_valid = ~np.isnan(y1_all.ravel())
    if not mask_valid.any() or mask_valid.sum() < 30:
        return None

    last_valid = np.where(mask_valid)[0].max()
    X_all      = X_all[:last_valid+1]
    y1_all     = y1_all[:last_valid+1]
    dir_all    = dir_all[:last_valid+1]
    mask_valid = mask_valid[:last_valid+1]
    B_all = df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].values.astype(float)[:last_valid+1]

    n      = len(X_all)
    cut    = int(0.8 * n)
    cut_lo = max(0, cut - EMBARGO_STEPS)
    cut_hi = min(n, cut + EMBARGO_STEPS)

    m_tr = np.zeros(n, dtype=bool); m_tr[:cut_lo] = True; m_tr &= mask_valid
    m_te = np.zeros(n, dtype=bool); m_te[cut_hi:] = True; m_te &= mask_valid

    if not m_tr.any() or not m_te.any():
        return None

    x_scaler  = MinMaxScaler().fit(X_all[m_tr])
    b_scaler  = MinMaxScaler().fit(B_all[m_tr])
    y1_scaler = StandardScaler().fit(y1_all[m_tr])

    X  = np.clip(x_scaler.transform(X_all),  -500.0, 500.0).astype(np.float32)
    B  = np.clip(b_scaler.transform(B_all),  -500.0, 500.0).astype(np.float32)
    y1 = y1_scaler.transform(y1_all)

    pack_tr, idxs_tr = make_sequences_masked(X, {"y1": y1, "dir": dir_all.reshape(-1,1), "B": B}, seq_len, m_tr)
    pack_te, idxs_te = make_sequences_masked(X, {"y1": y1, "dir": dir_all.reshape(-1,1), "B": B}, seq_len, m_te)

    if pack_tr["X"].shape[0] == 0 or pack_te["X"].shape[0] == 0:
        return None

    y1_mu, y1_sd = pack_tr["y1"].mean(), pack_tr["y1"].std() + 1e-9
    int_tr = np.tanh((pack_tr["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    int_te = np.tanh((pack_te["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    dir_tr = pack_tr["dir"].astype(np.float32)
    dir_te = apply_label_smoothing(pack_te["dir"])

    return {
        "X_tr_seq": pack_tr["X"], "X_te_seq": pack_te["X"],
        "y1_tr_seq": pack_tr["y1"], "y1_te_seq": pack_te["y1"],
        "dir_tr_seq": dir_tr, "dir_te_seq": dir_te,
        "int_tr_seq": int_tr, "int_te_seq": int_te,
        "B_tr_seq": pack_tr["B"], "B_te_seq": pack_te["B"],
        "x_scaler": x_scaler, "b_scaler": b_scaler, "y1_scaler": y1_scaler,
        "anchor_idx": cut + seq_len - 1,
        "orig_idxs_tr": np.array(idxs_tr),
        "orig_idxs_te": np.array(idxs_te)
    }


def prepare_daily_arrays(df, feature_cols, seq_len, daily_window=60):
    min_rows = seq_len + daily_window
    if len(df) < min_rows:
        log.info(f"  [Daily FT] Not enough rows (need {min_rows}, have {len(df)}) — skipping.")
        return None
    df_recent = df.tail(max(min_rows, 200)).reset_index(drop=True)
    pack = prepare_single_stock_arrays(df_recent, feature_cols, seq_len)
    if pack is None:
        log.info(f"  [Daily FT] prepare_single_stock_arrays returned None — skipping.")
    return pack


# ---------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------
def calculate_verdict(dir_probs, intensity_values, price_changes):
    avg_dir    = np.mean(dir_probs)
    avg_int    = np.mean(np.abs(intensity_values))
    avg_change = np.mean(price_changes)
    if avg_dir > 0.65 and avg_change > 1.0:
        return 'BULLISH', min(95, 60 + avg_int * 30), 'Strong upward momentum with high conviction'
    elif avg_dir < 0.35 and avg_change < -1.0:
        return 'BEARISH', min(95, 60 + avg_int * 30), 'Strong downward pressure with high conviction'
    elif avg_dir > 0.55 and avg_change > 0.3:
        return 'BULLISH', min(75, 50 + avg_int * 20), 'Moderate upward trend with positive signals'
    elif avg_dir < 0.45 and avg_change < -0.3:
        return 'BEARISH', min(75, 50 + avg_int * 20), 'Moderate downward trend with negative signals'
    else:
        return 'NEUTRAL', 40 + 20 * min(1.0, avg_int), 'Mixed signals with no clear directional bias'


# ---------------------------------------------------------------
# Evaluation (metrics only, no chart output unless ENABLE_VISUALS)
# ---------------------------------------------------------------
def evaluate_holdout_close(df, pack, model, vol_idx, symbol):
    if pack["X_te_seq"].shape[0] < 10:
        log.info(f"  [Abort Eval] {symbol}: n_val={pack['X_te_seq'].shape[0]} too small.")
        return {}
    preds = model.predict(
        {"price_seq":   pack["X_te_seq"],
         "break_feats": pack["B_te_seq"],
         "vol_level":   np.ones((pack["X_te_seq"].shape[0], 1), dtype=np.int32) * vol_idx},
        verbose=0
    )
    r1_pred     = preds[0]
    r1_pred_inv = pack["y1_scaler"].inverse_transform(r1_pred).ravel()
    y_true_inv  = pack["y1_scaler"].inverse_transform(pack["y1_te_seq"]).ravel()

    pred_close, true_close = [], []
    for k in range(len(r1_pred_inv)):
        actual_day_idx = pack["orig_idxs_te"][k]
        prev_day_idx   = actual_day_idx - 1
        if prev_day_idx < 0 or actual_day_idx >= len(df):
            continue
        base   = df.iloc[prev_day_idx]["Close"]
        actual = df.iloc[actual_day_idx]["Close"]
        pred_close.append(base * np.exp(r1_pred_inv[k]))
        true_close.append(actual)

    if not true_close:
        return {}

    mae  = mean_absolute_error(true_close, pred_close)
    rmse = np.sqrt(mean_squared_error(true_close, pred_close))
    r2   = r2_score(true_close, pred_close)

    dir_prob     = preds[1].ravel()
    neutral_band = 0.0015
    dir_label    = np.full_like(y_true_inv, -1, dtype=int)
    dir_label[y_true_inv >  neutral_band] = 1
    dir_label[y_true_inv < -neutral_band] = 0
    pred_dir = np.full_like(dir_prob, -1, dtype=int)
    pred_dir[dir_prob > 0.55] = 1
    pred_dir[dir_prob < 0.45] = 0
    mask = (dir_label != -1) & (pred_dir != -1)
    hit  = (pred_dir[mask] == dir_label[mask]).mean() * 100 if mask.sum() > 0 else 0.0

    log.info(f"\n{'='*60}")
    log.info(f"HOLDOUT METRICS: {symbol}")
    log.info(f"  MAE: Rs.{mae:.2f}  |  RMSE: Rs.{rmse:.2f}  |  R2: {r2:.4f}")
    log.info(f"  1-Day Direction Hit-Rate: {hit:.2f}%")
    log.info(f"{'='*60}\n")

    # Only produce charts if ENABLE_VISUALS is explicitly True
    if ENABLE_VISUALS:
        x_range = range(len(true_close))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(x_range, true_close, linewidth=2.5, label='Actual', marker='o', markersize=3, alpha=0.8)
        ax1.plot(x_range, pred_close, linewidth=2.5, label='Predicted', marker='s', markersize=3, alpha=0.8, linestyle='--')
        ax1.set_title(f'{symbol} - Actual vs Predicted (Holdout)', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Price (Rs.)', fontsize=12); ax1.legend()
        ax1.grid(True, alpha=0.3)
        accuracy_per_sample = (pred_dir == dir_label).astype(float) * 100
        ax2.bar(range(len(accuracy_per_sample)), accuracy_per_sample, alpha=0.7, edgecolor='black')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=2)
        ax2.axhline(hit, color='purple', linestyle='-', linewidth=2, label=f'Avg {hit:.1f}%')
        ax2.set_title('Direction Accuracy per Sample', fontsize=13)
        ax2.set_ylim([0, 105]); ax2.legend()
        plt.tight_layout(); plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2, "dir_hit_rate": hit}


# ---------------------------------------------------------------
# 1-Day forecast
# ---------------------------------------------------------------
def forecast_1d(df_in, model, feature_cols, b_scaler, x_scaler, y1_scaler, vol_idx, seq_len):
    if len(df_in) < seq_len:
        return None
    last_close = float(df_in["Close"].iloc[-1])
    last_date  = pd.to_datetime(df_in["Date"].iloc[-1]) if "Date" in df_in.columns else pd.Timestamp.today()

    feats = df_in[feature_cols].values
    X_seq = np.clip(x_scaler.transform(feats[-seq_len:]), -500.0, 500.0).reshape(
        1, seq_len, len(feature_cols)).astype(np.float32)
    B_seq = np.clip(b_scaler.transform(
        df_in[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].iloc[-1].values.reshape(1, -1)
    ), -500.0, 500.0).astype(np.float32)
    vol_arr = np.array([[vol_idx]], dtype=np.int32)

    preds   = model.predict({"price_seq": X_seq, "break_feats": B_seq, "vol_level": vol_arr}, verbose=0)
    r1_pred = preds[0]; dirp = preds[1]; inten = preds[2]

    next_ret = float(y1_scaler.inverse_transform(r1_pred)[0, 0])
    next_ret = np.clip(next_ret, -0.20, 0.20)
    if np.isnan(next_ret) or np.isinf(next_ret):
        next_ret = 0.0

    return {
        "date":        (last_date + BDay(1)).date(),
        "pred_close":  last_close * np.exp(next_ret),
        "pred_logret": next_ret,
        "dir_prob":    float(dirp[0, 0]),
        "intensity":   float(inten[0, 0]),
        "last_close":  last_close
    }


def plot_forecast_1d(df_in, fc, symbol):
    """Only called when ENABLE_VISUALS is True."""
    if not ENABLE_VISUALS:
        return
    hist        = df_in.tail(60)
    hist_dates  = pd.to_datetime(hist["Date"]) if "Date" in hist.columns else range(len(hist))
    hist_prices = hist["Close"].values
    plt.figure(figsize=(14, 6))
    plt.plot(hist_dates, hist_prices, label="Historical", marker='o', markersize=2)
    last_date = hist_dates.iloc[-1] if hasattr(hist_dates, "iloc") else hist_dates[-1]
    plt.plot([last_date, pd.to_datetime(fc["date"])],
             [fc["last_close"], fc["pred_close"]],
             label="1-Day Forecast", marker='D', linestyle='--')
    chg = (fc["pred_close"] - fc["last_close"]) / fc["last_close"] * 100
    plt.title(f"{symbol} | 1-Day Forecast | dir={fc['dir_prob']:.2f} | int={fc['intensity']:.2f}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.show()
    log.info(f"\n{symbol} next day ({fc['date']}): Rs.{fc['pred_close']:.2f} ({chg:+.2f}%) "
             f"| dir_prob={fc['dir_prob']:.3f}")


# ---------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------
def build_tabular_dataset_1d(df, feature_cols):
    X    = df[feature_cols]
    y    = df["LogRet"].shift(-1)
    mask = (~X.isna().any(axis=1)) & y.notna()
    return X[mask].values, y[mask].values

def train_lgbm_1d(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        objective="regression", random_state=42
    )
    model.fit(X, y)
    return model

def train_xgb_1d(X, y):
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 6, "eta": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "objective": "reg:squarederror",
        "eval_metric": "rmse", "seed": 42
    }
    return xgb.train(params, dtrain, num_boost_round=500)


# ---------------------------------------------------------------
# Ensemble signal
# ---------------------------------------------------------------
def ensemble_next_day_signal(df, feature_cols, meta, dl_model,
                             lgbm=None, xgb_model=None,
                             ret_threshold=0.003, prob_band=(0.48, 0.52)):
    seq_len   = meta["seq_len"]
    x_scaler  = meta["x_scaler"]
    b_scaler  = meta["b_scaler"]
    y1_scaler = meta["y1_scaler"]
    vol_idx   = meta["vol_idx"]

    if len(df) < seq_len:
        return None

    feats   = df[feature_cols].values
    X_seq   = np.clip(x_scaler.transform(feats[-seq_len:]), -500.0, 500.0).reshape(
        1, seq_len, len(feature_cols)).astype(np.float32)
    B_seq   = np.clip(b_scaler.transform(
        df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].iloc[-1].values.reshape(1, -1)
    ), -500.0, 500.0).astype(np.float32)
    vol_arr = np.array([[vol_idx]], dtype=np.int32)

    preds    = dl_model.predict({"price_seq": X_seq, "break_feats": B_seq, "vol_level": vol_arr}, verbose=0)
    dir_prob = float(preds[1][0, 0])
    inten    = float(preds[2][0, 0])
    dl_ret   = float(y1_scaler.inverse_transform(preds[0])[0, 0])

    row      = df[feature_cols].iloc[[-1]].values
    lgbm_ret = float(lgbm.predict(row)[0]) if lgbm is not None else None
    xgb_ret  = None
    if xgb_model is not None:
        xgb_ret = float(xgb_model.predict(xgb.DMatrix(row))[0])

    rets    = [dl_ret];    weights = [0.6]
    if lgbm_ret is not None: rets.append(lgbm_ret); weights.append(0.25)
    if xgb_ret  is not None: rets.append(xgb_ret);  weights.append(0.15)
    weights   = np.array(weights) / np.sum(weights)
    final_ret = float(np.dot(weights, np.array(rets)))

    last_close   = float(df["Close"].iloc[-1])
    target_price = last_close * np.exp(final_ret)

    low, high = prob_band
    reasons   = []
    if low < dir_prob < high:          reasons.append("DL dir prob near 0.5")
    if abs(final_ret) < ret_threshold: reasons.append("Ensemble expected move too small")
    signs = [np.sign(dl_ret)]
    if lgbm_ret is not None: signs.append(np.sign(lgbm_ret))
    if xgb_ret  is not None: signs.append(np.sign(xgb_ret))
    if len(set(signs)) > 1:            reasons.append("Model disagreement (DL vs trees)")

    action     = "NO_TRADE" if reasons else ("LONG" if final_ret > 0 else "SHORT")
    confidence = (max(0.0, 1.0 - len(reasons) * 0.25) if reasons
                  else min(1.0, 0.6 + abs(inten) * 0.3))

    return {
        "action": action, "target_price": target_price,
        "expected_ret_pct": final_ret * 100,
        "dl_ret": dl_ret * 100,
        "lgbm_ret": None if lgbm_ret is None else lgbm_ret * 100,
        "xgb_ret":  None if xgb_ret  is None else xgb_ret  * 100,
        "dir_prob": dir_prob, "intensity": inten,
        "confidence": confidence, "no_trade_reasons": reasons
    }


# ---------------------------------------------------------------
# ===== CSV OUTPUT HELPERS ======================================
# Master predictions CSV and volatility-wise ranked reports.
# These functions are called after all stocks in a run are processed.
# ---------------------------------------------------------------

def save_master_predictions_csv(all_predictions, cutoff_date):
    """
    Save a master CSV of predictions for all processed stocks.
    File: Outputs/Master_Predictions/master_predictions_<cutoff_date>.csv
    Each row = one stock prediction with all available fields.
    """
    if not all_predictions:
        log.info("[CSV] No predictions to save.")
        return
    out_dir = os.path.join(OUTPUT_DIR, "Master_Predictions")
    os.makedirs(out_dir, exist_ok=True)
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    fpath = os.path.join(out_dir, f"master_predictions_{cutoff_str}.csv")
    df_out = pd.DataFrame(all_predictions)
    df_out.to_csv(fpath, index=False)
    log.info(f"[CSV] Master predictions saved: {fpath}  ({len(df_out)} rows)")
    return fpath

def build_ranked_volatility_reports(all_predictions, cutoff_date, top_n=20):
    """
    For each volatility class, create top-N and bottom-N CSVs ranked by predicted_return.
    Also saves a combined ranked_summary CSV.
    Files: Outputs/Ranked_Reports/<cutoff_date>/top20_<VOL>_<date>.csv etc.
    """
    if not all_predictions:
        return
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    report_dir = os.path.join(OUTPUT_DIR, "Ranked_Reports", cutoff_str)
    os.makedirs(report_dir, exist_ok=True)

    df_all = pd.DataFrame(all_predictions)
    if "predicted_return" not in df_all.columns or "volatility_class" not in df_all.columns:
        log.info("[Ranked] Missing required columns; skipping ranked reports.")
        return

    df_all["predicted_return"] = pd.to_numeric(df_all["predicted_return"], errors="coerce")
    df_all = df_all.dropna(subset=["predicted_return"])

    rank_cols = ["rank", "symbol", "sector", "volatility_class", "bucket_name",
                 "last_close", "predicted_price", "predicted_return",
                 "predicted_direction", "confidence"]
    rank_cols = [c for c in rank_cols if c in df_all.columns]

    summary_rows = []
    for vol_class in df_all["volatility_class"].unique():
        sub = df_all[df_all["volatility_class"] == vol_class].copy()
        sub = sub.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        sub["rank"] = sub.index + 1

        top = sub.head(top_n)[rank_cols]
        top.to_csv(os.path.join(report_dir, f"top{top_n}_{vol_class}_{cutoff_str}.csv"), index=False)

        bottom = sub.tail(top_n).sort_values("predicted_return").reset_index(drop=True)
        bottom["rank"] = bottom.index + 1
        bottom = bottom[rank_cols]
        bottom.to_csv(os.path.join(report_dir, f"bottom{top_n}_{vol_class}_{cutoff_str}.csv"), index=False)
        log.info(f"[Ranked] {vol_class}: top/bottom{top_n} saved.")

        summary_rows.append(sub)

    if summary_rows:
        df_summary = pd.concat(summary_rows, ignore_index=True)
        df_summary = df_summary.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        df_summary["overall_rank"] = df_summary.index + 1
        df_summary.to_csv(os.path.join(report_dir, f"ranked_summary_{cutoff_str}.csv"), index=False)
        log.info(f"[Ranked] Overall summary saved: ranked_summary_{cutoff_str}.csv")


# ---------------------------------------------------------------
# ===== STAGE RUNNERS ===========================================
# Each stage function is called explicitly from main() based on
# RUN_STAGE. They do NOT call each other — higher stages are
# never automatically triggered by lower-stage runs.
# ---------------------------------------------------------------

def run_yearly_pretrain(buckets, bucket_feature_union, cutoff_date):
    """
    Stage 1: Yearly pretraining per bucket.
    Uses full history up to cutoff_date.
    Saves encoder weights + metadata to Models/Yearly_Pretrained/<cutoff_date>/<bucket>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[YEARLY PRETRAIN] cutoff={cutoff_date.date()}")
    log.info(f"{'='*60}")
    cutoff_str = str(cutoff_date.date())
    bucket_encoder_path = {}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        out_dir    = os.path.join(STAGE_DIRS["yearly_pretrain"], cutoff_str, bucket_tag)
        os.makedirs(out_dir, exist_ok=True)

        enc_path = os.path.join(out_dir, "encoder.weights.h5")
        if os.path.exists(enc_path) and os.path.exists(os.path.join(out_dir, "stage_meta.json")):
            log.info(f"[Skip pretrain] ({vol_level}/{sector}) - checkpoint exists for {cutoff_str}")
            bucket_encoder_path[(vol_level, sector)] = enc_path
            continue

        log.info(f"\n[Pretrain] ({vol_level}/{sector}) | {len(items)} stocks | {len(feature_cols_union)} features")
        enc_path = pretrain_bucket(items, feature_cols_union, vol_level, sector, out_dir,
                                   cutoff_date=cutoff_date)
        if enc_path:
            bucket_encoder_path[(vol_level, sector)] = enc_path

    return bucket_encoder_path


def run_monthly_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 2: Monthly fine-tuning per stock.
    Uses MONTHLY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from yearly pretrained encoder (parent checkpoint).
    Saves to Models/Monthly_Finetuned/<cutoff_date>/<bucket>/<symbol>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[MONTHLY FINETUNE] cutoff={cutoff_date.date()}  window={MONTHLY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["monthly_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            # Apply monthly window
            df = apply_stage_window(df_full, "monthly_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip monthly FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Monthly FT] {sym}  (vol={vol_level}, sector={sector}) rows={len(df)}")
            pack = prepare_single_stock_arrays(df, feature_cols_union, SEQ_LEN)
            if pack is None:
                log.info(f"  [Skip monthly FT] {sym}: insufficient data for train/val split")
                continue

            n_features = pack["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            monthly_ckpt = os.path.join(out_dir, "best.weights.h5")
            if os.path.exists(monthly_ckpt):
                try:
                    model.load_weights(monthly_ckpt)
                    log.info(f"  [Resume] Loaded monthly weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from yearly pretrained encoder
                parent_path, parent_meta = resolve_parent_checkpoint(
                    "monthly_finetune", bucket_tag, symbol=None)
                if parent_path and os.path.isfile(parent_path):
                    try:
                        model.get_layer("advanced_encoder").load_weights(parent_path)
                        log.info(f"  [Parent] Loaded yearly encoder: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent encoder: {e}")

            vol_tr = np.full((pack["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te = np.full((pack["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_MONTHLY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_FT_M)

            model.fit(
                {"price_seq": pack["X_tr_seq"], "break_feats": pack["B_tr_seq"], "vol_level": vol_tr},
                {"r1": pack["y1_tr_seq"], "dir": pack["dir_tr_seq"], "int": pack["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack["X_te_seq"], "break_feats": pack["B_te_seq"], "vol_level": vol_te},
                    {"r1": pack["y1_te_seq"], "dir": pack["dir_te_seq"], "int": pack["int_te_seq"]}
                ),
                epochs=EPOCHS_MONTHLY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
                    ModelCheckpoint(monthly_ckpt, monitor="val_loss", save_best_only=True,
                                    save_weights_only=True, verbose=0)
                ],
                verbose=1
            )

            if os.path.exists(monthly_ckpt):
                model.load_weights(monthly_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            parent_path_str = str(resolve_parent_checkpoint("monthly_finetune", bucket_tag)[0])
            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "monthly_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack["x_scaler"], "b_scaler": pack["b_scaler"],
                "y1_scaler": pack["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            log.info(f"  [Monthly FT] {sym}: saved -> {model_path}")

            # Forecast for CSV output
            fc = forecast_1d(df, model, feature_cols_union,
                             pack["b_scaler"], pack["x_scaler"], pack["y1_scaler"],
                             vol_idx, SEQ_LEN)
            if fc is not None:
                direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
                all_predictions.append({
                    "prediction_date":       str(fc["date"]),
                    "cutoff_date":           str(cutoff_date.date()),
                    "symbol":                sym,
                    "sector":                sector,
                    "volatility_class":      vol_level,
                    "bucket_name":           bucket_tag,
                    "last_close":            round(fc["last_close"], 2),
                    "predicted_price":       round(fc["pred_close"], 2),
                    "predicted_return":      round(fc["pred_logret"] * 100, 4),
                    "predicted_direction":   direction,
                    "dir_prob":              round(fc["dir_prob"], 4),
                    "intensity":             round(fc["intensity"], 4),
                    "model_stage_used":      "monthly_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })


def run_weekly_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 3: Weekly fine-tuning per stock.
    Uses WEEKLY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from monthly finetuned checkpoint (parent). Falls back to yearly if monthly missing.
    Saves to Models/Weekly_Finetuned/<cutoff_date>/<bucket>/<symbol>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[WEEKLY FINETUNE] cutoff={cutoff_date.date()}  window={WEEKLY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["weekly_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            df = apply_stage_window(df_full, "weekly_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip weekly FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Weekly FT] {sym}  (vol={vol_level}, sector={sector}) rows={len(df)}")
            pack = prepare_single_stock_arrays(df, feature_cols_union, SEQ_LEN)
            if pack is None:
                log.info(f"  [Skip weekly FT] {sym}: insufficient data")
                continue

            n_features = pack["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            weekly_ckpt = os.path.join(out_dir, "best.weights.h5")
            parent_path_str = ""
            if os.path.exists(weekly_ckpt):
                try:
                    model.load_weights(weekly_ckpt)
                    log.info(f"  [Resume] Loaded weekly weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from monthly (symbol-level first, then bucket-level encoder)
                parent_path, parent_meta = resolve_parent_checkpoint(
                    "weekly_finetune", bucket_tag, symbol=sym)
                if parent_path:
                    parent_path_str = str(parent_path)
                    try:
                        if os.path.isdir(parent_path):
                            parent_model = tf.keras.models.load_model(
                                parent_path,
                                custom_objects={"GatingLayer": GatingLayer,
                                                "PositionalEncoding": PositionalEncoding}
                            )
                            model.get_layer("advanced_encoder").set_weights(
                                parent_model.get_layer("advanced_encoder").get_weights())
                        elif os.path.isfile(parent_path):
                            model.get_layer("advanced_encoder").load_weights(parent_path)
                        log.info(f"  [Parent] Loaded monthly checkpoint: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent: {e}")

            vol_tr = np.full((pack["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te = np.full((pack["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_WEEKLY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_FT)

            model.fit(
                {"price_seq": pack["X_tr_seq"], "break_feats": pack["B_tr_seq"], "vol_level": vol_tr},
                {"r1": pack["y1_tr_seq"], "dir": pack["dir_tr_seq"], "int": pack["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack["X_te_seq"], "break_feats": pack["B_te_seq"], "vol_level": vol_te},
                    {"r1": pack["y1_te_seq"], "dir": pack["dir_te_seq"], "int": pack["int_te_seq"]}
                ),
                epochs=EPOCHS_WEEKLY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
                    ModelCheckpoint(weekly_ckpt, monitor="val_loss", save_best_only=True,
                                    save_weights_only=True, verbose=0)
                ],
                verbose=1
            )

            if os.path.exists(weekly_ckpt):
                model.load_weights(weekly_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "weekly_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack["x_scaler"], "b_scaler": pack["b_scaler"],
                "y1_scaler": pack["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            # Evaluate (metrics only)
            if pack["X_te_seq"].shape[0] >= 10:
                evaluate_holdout_close(df, pack, model, vol_idx, sym)

            # Tree models
            X_tab, y_tab = build_tabular_dataset_1d(df, feature_cols_union)
            lgbm_model, xgb_model = None, None
            if len(X_tab) > 100:
                lgbm_model = train_lgbm_1d(X_tab, y_tab)
                xgb_model  = train_xgb_1d(X_tab, y_tab)
                tree_dir   = os.path.join(out_dir, "tree_models")
                os.makedirs(tree_dir, exist_ok=True)
                joblib.dump(lgbm_model, os.path.join(tree_dir, "lgbm_1d.pkl"))
                xgb_model.save_model(os.path.join(tree_dir, "xgb_1d.json"))
                log.info(f"  [Tree] LGBM + XGBoost saved for {sym}")

            # Forecast
            fc = forecast_1d(df, model, feature_cols_union,
                             pack["b_scaler"], pack["x_scaler"], pack["y1_scaler"],
                             vol_idx, SEQ_LEN)
            if fc is not None:
                direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
                all_predictions.append({
                    "prediction_date":       str(fc["date"]),
                    "cutoff_date":           str(cutoff_date.date()),
                    "symbol":                sym,
                    "sector":                sector,
                    "volatility_class":      vol_level,
                    "bucket_name":           bucket_tag,
                    "last_close":            round(fc["last_close"], 2),
                    "predicted_price":       round(fc["pred_close"], 2),
                    "predicted_return":      round(fc["pred_logret"] * 100, 4),
                    "predicted_direction":   direction,
                    "dir_prob":              round(fc["dir_prob"], 4),
                    "intensity":             round(fc["intensity"], 4),
                    "model_stage_used":      "weekly_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })


def run_daily_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 4: Daily fine-tuning per stock.
    Uses DAILY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from weekly finetuned checkpoint (parent). Falls back through hierarchy.
    Saves to Models/Daily_Finetuned/<cutoff_date>/<bucket>/<symbol>/

    Running this stage does NOT trigger weekly/monthly/yearly reruns.
    Appending new rows to the dataset will shift the cutoff when this
    stage is next explicitly run.
    """
    log.info(f"\n{'='*60}")
    log.info(f"[DAILY FINETUNE] cutoff={cutoff_date.date()}  window={DAILY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["daily_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            df = apply_stage_window(df_full, "daily_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip daily FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Daily FT] {sym}  (vol={vol_level}) rows={len(df)}")
            pack_daily = prepare_daily_arrays(df, feature_cols_union, SEQ_LEN, daily_window=60)
            if pack_daily is None or pack_daily["X_tr_seq"].shape[0] < 5:
                log.info(f"  [Skip daily FT] {sym}: insufficient sequences")
                continue

            n_features = pack_daily["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            daily_ckpt = os.path.join(out_dir, "best.weights.h5")
            parent_path_str = ""
            if os.path.exists(daily_ckpt):
                try:
                    model.load_weights(daily_ckpt)
                    log.info(f"  [Resume] Loaded daily weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from latest weekly checkpoint (symbol-level preferred)
                parent_path, _ = resolve_parent_checkpoint(
                    "daily_finetune", bucket_tag, symbol=sym)
                if parent_path:
                    parent_path_str = str(parent_path)
                    try:
                        if os.path.isdir(parent_path):
                            parent_model = tf.keras.models.load_model(
                                parent_path,
                                custom_objects={"GatingLayer": GatingLayer,
                                                "PositionalEncoding": PositionalEncoding}
                            )
                            model.set_weights(parent_model.get_weights())
                        elif os.path.isfile(parent_path):
                            model.load_weights(parent_path)
                        log.info(f"  [Parent] Loaded weekly checkpoint: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent: {e}")

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_DAILY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_DAILY)

            vol_tr_d = np.full((pack_daily["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te_d = np.full((pack_daily["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            has_daily_val = pack_daily["X_te_seq"].shape[0] > 0

            model.fit(
                {"price_seq": pack_daily["X_tr_seq"], "break_feats": pack_daily["B_tr_seq"], "vol_level": vol_tr_d},
                {"r1": pack_daily["y1_tr_seq"], "dir": pack_daily["dir_tr_seq"], "int": pack_daily["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack_daily["X_te_seq"], "break_feats": pack_daily["B_te_seq"], "vol_level": vol_te_d},
                    {"r1": pack_daily["y1_te_seq"], "dir": pack_daily["dir_te_seq"], "int": pack_daily["int_te_seq"]}
                ) if has_daily_val else None,
                epochs=EPOCHS_DAILY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss" if has_daily_val else "loss",
                                  patience=3, restore_best_weights=True),
                    ModelCheckpoint(daily_ckpt, monitor="val_loss" if has_daily_val else "loss",
                                    save_best_only=True, save_weights_only=True, verbose=0)
                ],
                verbose=0
            )

            if os.path.exists(daily_ckpt):
                model.load_weights(daily_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "daily_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack_daily["x_scaler"], "b_scaler": pack_daily["b_scaler"],
                "y1_scaler": pack_daily["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            # Forecast + collect for master CSV
            fc = forecast_1d(df, model, feature_cols_union,
                             pack_daily["b_scaler"], pack_daily["x_scaler"], pack_daily["y1_scaler"],
                             vol_idx, SEQ_LEN)
            if fc is not None:
                direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
                all_predictions.append({
                    "prediction_date":       str(fc["date"]),
                    "cutoff_date":           str(cutoff_date.date()),
                    "symbol":                sym,
                    "sector":                sector,
                    "volatility_class":      vol_level,
                    "bucket_name":           bucket_tag,
                    "last_close":            round(fc["last_close"], 2),
                    "predicted_price":       round(fc["pred_close"], 2),
                    "predicted_return":      round(fc["pred_logret"] * 100, 4),
                    "predicted_direction":   direction,
                    "dir_prob":              round(fc["dir_prob"], 4),
                    "intensity":             round(fc["intensity"], 4),
                    "model_stage_used":      "daily_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })


def run_predict_only(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    predict_only: load the best available checkpoint for each stock (daily > weekly > monthly > yearly)
    and run forecast without any training. Writes predictions to master CSV.
    """
    log.info(f"\n{'='*60}")
    log.info(f"[PREDICT ONLY] cutoff={cutoff_date.date()}")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    stage_priority = ["daily_finetune", "weekly_finetune", "monthly_finetune", "yearly_pretrain"]

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            df = filter_df_to_cutoff(df_full, cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            model       = None
            meta_pkl    = None
            stage_used  = "none"
            ckpt_path_used = ""

            for stage in stage_priority:
                ckpt_path, stage_meta = load_latest_successful_checkpoint(stage, bucket_tag, symbol=sym)
                if ckpt_path is None:
                    continue
                meta_path = os.path.join(os.path.dirname(ckpt_path), "meta.pkl")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, "rb") as f:
                        meta_pkl = pickle.load(f)
                    n_features = len(meta_pkl["feature_cols"])
                    candidate  = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)
                    if os.path.isdir(ckpt_path):
                        candidate = tf.keras.models.load_model(
                            ckpt_path,
                            custom_objects={"GatingLayer": GatingLayer,
                                            "PositionalEncoding": PositionalEncoding}
                        )
                    else:
                        candidate.load_weights(ckpt_path)
                    model = candidate
                    stage_used = stage
                    ckpt_path_used = str(ckpt_path)
                    break
                except Exception as e:
                    log.warning(f"  [Predict] {sym} {stage} load failed: {e}")
                    continue

            if model is None or meta_pkl is None:
                log.info(f"  [Skip predict] {sym}: no usable checkpoint found.")
                continue

            fc = forecast_1d(df, model, meta_pkl["feature_cols"],
                             meta_pkl["b_scaler"], meta_pkl["x_scaler"], meta_pkl["y1_scaler"],
                             meta_pkl["vol_idx"], SEQ_LEN)
            if fc is None:
                continue

            direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
            all_predictions.append({
                "prediction_date":       str(fc["date"]),
                "cutoff_date":           cutoff_str,
                "symbol":                sym,
                "sector":                sector,
                "volatility_class":      vol_level,
                "bucket_name":           bucket_tag,
                "last_close":            round(fc["last_close"], 2),
                "predicted_price":       round(fc["pred_close"], 2),
                "predicted_return":      round(fc["pred_logret"] * 100, 4),
                "predicted_direction":   direction,
                "dir_prob":              round(fc["dir_prob"], 4),
                "intensity":             round(fc["intensity"], 4),
                "model_stage_used":      stage_used,
                "parent_checkpoint_path": ckpt_path_used,
                "data_window_start":     "",
                "data_window_end":       str(cutoff_date.date()),
                "rows_used":             len(df),
            })


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    for d in [MODEL_DIR, OUTPUT_DIR] + list(STAGE_DIRS.values()):
        os.makedirs(d, exist_ok=True)

    # ── Auto-detect cutoff date ──────────────────────────────────
    # Each stage run detects the latest date currently in the dataset.
    # This means: update your CSV data, then run the stage you want.
    # No manual date entry required.
    if AUTO_DETECT_CUTOFF_DATE:
        run_cutoff_date = detect_latest_dataset_date(DATA_DIR)
    else:
        run_cutoff_date = pd.Timestamp.today().normalize()
    log.info(f"\nRUN_STAGE        = {RUN_STAGE}")
    log.info(f"RUN_CUTOFF_DATE  = {run_cutoff_date.date()}")
    log.info(f"ENABLE_VISUALS   = {ENABLE_VISUALS}")

    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    # ── Discover & load CSVs ─────────────────────────────────────
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {DATA_DIR}")
        return
    log.info(f"Found {len(csv_files)} CSV file(s).")

    # ── Load & bucket stocks ─────────────────────────────────────
    buckets      = {}
    total_stocks = 0

    for path in csv_files:
        sym_dfs = load_csv_by_symbol(path)
        for sym, df_raw in sym_dfs.items():
            # Apply cutoff filter so buckets only see data up to cutoff
            df_raw_cut = filter_df_to_cutoff(df_raw, run_cutoff_date)
            if len(df_raw_cut) < MIN_HISTORY_BARS:
                log.info(f"  [Skip] {sym}: only {len(df_raw_cut)} bars after cutoff filter")
                continue
            try:
                df, fcols, bcols = build_features_from_df(df_raw_cut)
            except Exception as e:
                log.warning(f"  [Skip] {sym}: feature build failed - {e}")
                continue

            vol_level, sector = assign_bucket(sym, df_raw_cut)
            buckets.setdefault((vol_level, sector), []).append((sym, df, fcols, bcols))
            total_stocks += 1

    log.info(f"\nTotal stocks loaded: {total_stocks}")
    buckets = merge_small_buckets(buckets, min_size=20)
    for (vol_level, sector), items in sorted(buckets.items()):
        log.info(f"  Bucket ({vol_level}, {sector}): {len(items)} stocks")

    # ── Union feature set per bucket ─────────────────────────────
    bucket_feature_union = {}
    for key, items in buckets.items():
        union = []
        for (sym, df, fcols, bcols) in items:
            for c in fcols:
                if c not in union:
                    union.append(c)
        bucket_feature_union[key] = union

    # ── Collect all predictions for CSV output ───────────────────
    all_predictions = []

    # ── Dispatch to requested stage ──────────────────────────────
    # Stages are INDEPENDENT. Running daily does NOT rerun weekly/monthly/yearly.
    # Each stage loads its parent checkpoint from the hierarchy.
    if RUN_STAGE == "yearly_pretrain":
        run_yearly_pretrain(buckets, bucket_feature_union, run_cutoff_date)

    elif RUN_STAGE == "monthly_finetune":
        run_monthly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif RUN_STAGE == "weekly_finetune":
        run_weekly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif RUN_STAGE == "daily_finetune":
        run_daily_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif RUN_STAGE == "predict_only":
        run_predict_only(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif RUN_STAGE == "full_pipeline":
        # Runs all 4 stages sequentially with the same cutoff date.
        # NOTE: This is the only case where higher stages are triggered automatically.
        log.info("[Full Pipeline] Running all 4 stages sequentially.")
        run_yearly_pretrain(buckets, bucket_feature_union, run_cutoff_date)
        run_monthly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)
        run_weekly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)
        run_daily_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    else:
        log.error(f"Unknown RUN_STAGE: '{RUN_STAGE}'. "
                  "Choose from: yearly_pretrain, monthly_finetune, weekly_finetune, "
                  "daily_finetune, predict_only, full_pipeline")
        return

    # ── Save master predictions CSV + ranked reports ─────────────
    if all_predictions:
        save_master_predictions_csv(all_predictions, run_cutoff_date)
        build_ranked_volatility_reports(all_predictions, run_cutoff_date, top_n=20)

    log.info(f"\n[Done] Stage '{RUN_STAGE}' completed. cutoff={run_cutoff_date.date()}")


if __name__ == "__main__":
    main()