"""
Advanced indicator functions.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.technical_indicators import (
    rsi,
    _ema,
    _wma,
    _hma,
    _kama,
    _macd,
    _tsi,
    _stochastic,
    _ultimate_oscillator,
    _connors_rsi,
    _fisher_transform,
)

from gainify_stock_predictor.features.calculus_features import _rolling_slope


EPS = 1e-9


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
