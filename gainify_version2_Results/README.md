# Gainify Quant Engine v2 — Public Tables

This folder contains a curated public CSV subset from the Gainify Quant Engine v2 backtest results.

Included CSVs:

- `metrics_summary.csv` — headline portfolio and trade metrics
- `equity_curve.csv` — daily public equity curve fields
- `rolling_drawdown.csv` — drawdown and rolling Sharpe series
- `yearly_performance.csv` — year-wise returns
- `regime_performance.csv` — regime-wise public performance summary
- `walkforward_splits.csv` — train/test windows with purge and embargo metadata
- `capacity_estimates.csv` — high-level liquidity/capacity summary
- `tradebook_public.csv` — sanitized tradebook without internal scores, ranks, feature states, or research diagnostics
- `top_5_trades.csv` — highest net PnL trades from the public tradebook
- `worst_5_trades.csv` — lowest net PnL trades from the public tradebook

Internal audit logs, daily ranking files, parameter profiles, and raw diagnostic outputs are intentionally not part of this public results subset.
