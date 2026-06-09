# Gainify Stock Predictor

A modular research-grade NSE stock prediction pipeline using advanced feature engineering, volatility-sector bucketing, staged transfer learning, and CNN-TFT-Attention models.

Important note: The original working files are preserved in legacy/ for reproducibility. The modular package under src/ progressively extracts and wraps the original logic.

## Architecture Overview

Gainify Stock Predictor is organized around a staged research pipeline:

- Data fetching modules update NSE stock and market-indicator datasets.
- Preprocessing modules normalize columns, clean numeric data, scale values, and build training sequences.
- Feature modules preserve the advanced feature-engineering pipeline extracted from the legacy implementation.
- Bucketing modules group stocks by volatility and sector behavior.
- Model modules define the CNN, TFT, attention, multitask heads, and tree-model helpers.
- Training modules run the yearly, monthly, weekly, and daily staged transfer-learning flow.
- Checkpoint modules manage the yearly -> monthly -> weekly -> daily hierarchy.
- Prediction and reporting modules generate ranked outputs from trained checkpoints.

## Folder Structure

```text
configs/                         Configuration modules
data/sample/                     Small sample data allowed in Git
legacy/                          Original working files for reproducibility
notebooks/                       Research notebooks
scripts/                         CLI entry points for fetching, training, and prediction
src/gainify_stock_predictor/     Modular package source
tests/                           Regression and structure tests
tools/                           Extraction and maintenance tools
```

Large runtime datasets, trained models, generated outputs, logs, caches, and local environments are ignored from GitHub.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` if you want to override project paths locally.

## Fetch Data

```powershell
python scripts\fetch_data.py
```

To update existing data:

```powershell
python scripts\update_data.py
```

## Train

Run yearly pretraining:

```powershell
python scripts\train_yearly.py
```

Run monthly fine-tuning:

```powershell
python scripts\train_monthly.py
```

Run weekly fine-tuning:

```powershell
python scripts\train_weekly.py
```

Run daily fine-tuning:

```powershell
python scripts\train_daily.py
```

Run the full staged pipeline:

```powershell
python scripts\run_full_pipeline.py
```

## Predict

```powershell
python scripts\predict.py
```

## Tests

```powershell
python -m compileall .
python -m pytest -q
```

## Ignored From GitHub

The `.gitignore` excludes Python caches, pytest caches, build artifacts, virtual environments, local `.env` files, logs, raw and processed datasets, generated stock CSVs, trained model/checkpoint files, generated outputs, runtime state files, and notebook checkpoints. Small sample CSVs under `data/sample/` are allowed.

## Disclaimer

This project is for research only and is not financial advice. Stock-market prediction is uncertain, and any trading or investment decision is your responsibility.
