"""
Full feature pipeline.

Function body physically extracted from legacy/yearly.py.
Original feature-engineering sequence is preserved.
"""

import numpy as np
import pandas as pd

from configs.bucket_config import IPO_RECENT_DAYS
from configs.market_config import COL_MAP, SECTOR_INDEX_INTERNAL

from gainify_stock_predictor.features.technical_indicators import (
    rsi,
    nw_kernel_smooth,
)

from gainify_stock_predictor.features.advanced_indicators import add_raw_advanced_features
from gainify_stock_predictor.features.statistical_features import add_raw_statistics_and_calculus
from gainify_stock_predictor.features.regime_features import (
    add_family_ecosystem_features,
    add_indicator_internal_dots,
    add_cross_family_dot_connections,
    add_final_market_evidence_scores,
)
from gainify_stock_predictor.features.candlestick_features import add_price_volume_structure_dots
from gainify_stock_predictor.features.probability_features import add_probability_ecosystem_features


EPS = 1e-9


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


def run_feature_pipeline(df_raw):
    """
    Clean modular alias for scripts.
    """
    return build_features_from_df(df_raw)
