"""
Regime, ecosystem, and dot-connection features.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.statistical_features import _rolling_z


EPS = 1e-9


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


def add_regime_features(df):
    """
    Runs the original regime/ecosystem feature sequence.
    """
    df = add_family_ecosystem_features(df)
    df = add_indicator_internal_dots(df)
    df = add_cross_family_dot_connections(df)
    df = add_final_market_evidence_scores(df)
    return df
