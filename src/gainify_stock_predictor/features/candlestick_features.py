"""
Candlestick and price-volume structure features.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.calculus_features import _consecutive_condition_count


EPS = 1e-9


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


def add_candlestick_features(df):
    """
    Compatibility alias for modular naming.
    """
    return add_price_volume_structure_dots(df)
