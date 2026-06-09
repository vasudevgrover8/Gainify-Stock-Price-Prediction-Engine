"""
Probability ecosystem features.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.statistical_features import (
    _safe_div,
    _rolling_z,
    _rolling_percentile,
)


EPS = 1e-9


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
