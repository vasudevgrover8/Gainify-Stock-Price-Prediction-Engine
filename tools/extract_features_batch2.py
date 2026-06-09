"""
Extract feature functions from legacy/yearly.py into modular feature files.

This does NOT rewrite your logic.
It copies exact function bodies from legacy/yearly.py using Python AST.
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_YEARLY = PROJECT_ROOT / "legacy" / "yearly.py"
FEATURE_DIR = PROJECT_ROOT / "src" / "gainify_stock_predictor" / "features"


source = LEGACY_YEARLY.read_text(encoding="utf-8")
tree = ast.parse(source)


def extract_function(name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise RuntimeError(f"Could not extract source for function: {name}")
            return text.strip() + "\n"
    raise RuntimeError(f"Function not found in legacy/yearly.py: {name}")


def extract_existing(function_names):
    blocks = []
    missing = []

    for name in function_names:
        try:
            blocks.append(extract_function(name))
        except RuntimeError:
            missing.append(name)

    if missing:
        print("[WARN] Missing functions:", missing)

    return "\n\n".join(blocks)


def write_file(path: Path, header: str, function_names, footer: str = ""):
    body = extract_existing(function_names)

    content = header.rstrip() + "\n\n\n" + body.rstrip() + "\n"

    if footer.strip():
        content += "\n\n" + footer.strip() + "\n"

    path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {path.relative_to(PROJECT_ROOT)}")


# ------------------------------------------------------------------
# 1. statistical_features.py
# Keep the exact original add_raw_statistics_and_calculus() here.
# This preserves the original combined statistics+calculus function.
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "statistical_features.py",
    '''"""
Statistical and combined statistics-calculus feature helpers.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.calculus_features import (
    _rolling_slope,
    _rolling_linear_r2,
    _rolling_quadratic_curvature,
    _consecutive_condition_count,
)


EPS = 1e-9
''',
    [
        "_safe_div",
        "_clip_series",
        "_rolling_z",
        "_robust_z",
        "_rolling_percentile",
        "_rolling_entropy",
        "_rolling_autocorr",
        "_hurst_approx",
        "_variance_ratio",
        "add_raw_statistics_and_calculus",
    ],
    '''
def add_statistical_features(df):
    """
    Compatibility alias.
    Calls the original full add_raw_statistics_and_calculus().
    """
    return add_raw_statistics_and_calculus(df)
'''
)


# ------------------------------------------------------------------
# 2. advanced_indicators.py
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "advanced_indicators.py",
    '''"""
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
''',
    [
        "_true_range",
        "_adx_dmi",
        "_aroon",
        "_supertrend",
        "_choppiness_index",
        "_obv",
        "_cmf",
        "_mfi",
        "_klinger",
        "_ease_of_movement",
        "_rolling_beta_alpha_corr",
        "_weekly_features",
        "add_raw_advanced_features",
    ],
)


# ------------------------------------------------------------------
# 3. candlestick_features.py
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "candlestick_features.py",
    '''"""
Candlestick and price-volume structure features.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.calculus_features import _consecutive_condition_count


EPS = 1e-9
''',
    [
        "add_price_volume_structure_dots",
    ],
    '''
def add_candlestick_features(df):
    """
    Compatibility alias for modular naming.
    """
    return add_price_volume_structure_dots(df)
'''
)


# ------------------------------------------------------------------
# 4. regime_features.py
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "regime_features.py",
    '''"""
Regime, ecosystem, and dot-connection features.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.statistical_features import _rolling_z


EPS = 1e-9
''',
    [
        "_squash",
        "add_family_ecosystem_features",
        "add_indicator_internal_dots",
        "add_cross_family_dot_connections",
        "add_final_market_evidence_scores",
    ],
    '''
def add_regime_features(df):
    """
    Runs the original regime/ecosystem feature sequence.
    """
    df = add_family_ecosystem_features(df)
    df = add_indicator_internal_dots(df)
    df = add_cross_family_dot_connections(df)
    df = add_final_market_evidence_scores(df)
    return df
'''
)


# ------------------------------------------------------------------
# 5. probability_features.py
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "probability_features.py",
    '''"""
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
''',
    [
        "add_probability_ecosystem_features",
    ],
)


# ------------------------------------------------------------------
# 6. feature_pipeline.py
# ------------------------------------------------------------------
write_file(
    FEATURE_DIR / "feature_pipeline.py",
    '''"""
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
''',
    [
        "build_features_from_df",
    ],
    '''
def run_feature_pipeline(df_raw):
    """
    Clean modular alias for scripts.
    """
    return build_features_from_df(df_raw)
'''
)


# ------------------------------------------------------------------
# 7. Update features/__init__.py
# ------------------------------------------------------------------
init_content = '''from gainify_stock_predictor.features.technical_indicators import (
    rsi,
    nw_kernel_smooth,
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

from gainify_stock_predictor.features.advanced_indicators import (
    _true_range,
    _adx_dmi,
    _aroon,
    _supertrend,
    _choppiness_index,
    _obv,
    _cmf,
    _mfi,
    _klinger,
    _ease_of_movement,
    _rolling_beta_alpha_corr,
    _weekly_features,
    add_raw_advanced_features,
)

from gainify_stock_predictor.features.statistical_features import (
    _safe_div,
    _clip_series,
    _rolling_z,
    _robust_z,
    _rolling_percentile,
    _rolling_entropy,
    _rolling_autocorr,
    _hurst_approx,
    _variance_ratio,
    add_statistical_features,
    add_raw_statistics_and_calculus,
)

from gainify_stock_predictor.features.calculus_features import (
    _rolling_slope,
    _rolling_linear_r2,
    _rolling_quadratic_curvature,
    _consecutive_condition_count,
    add_calculus_features,
)

from gainify_stock_predictor.features.probability_features import (
    add_probability_ecosystem_features,
)

from gainify_stock_predictor.features.relative_strength_features import (
    add_relative_strength_features,
)

from gainify_stock_predictor.features.candlestick_features import (
    add_price_volume_structure_dots,
    add_candlestick_features,
)

from gainify_stock_predictor.features.regime_features import (
    _squash,
    add_family_ecosystem_features,
    add_indicator_internal_dots,
    add_cross_family_dot_connections,
    add_final_market_evidence_scores,
    add_regime_features,
)

from gainify_stock_predictor.features.feature_pipeline import (
    build_features_from_df,
    run_feature_pipeline,
)


__all__ = [
    "rsi",
    "nw_kernel_smooth",
    "_ema",
    "_wma",
    "_hma",
    "_kama",
    "_macd",
    "_tsi",
    "_stochastic",
    "_ultimate_oscillator",
    "_connors_rsi",
    "_fisher_transform",

    "_true_range",
    "_adx_dmi",
    "_aroon",
    "_supertrend",
    "_choppiness_index",
    "_obv",
    "_cmf",
    "_mfi",
    "_klinger",
    "_ease_of_movement",
    "_rolling_beta_alpha_corr",
    "_weekly_features",
    "add_raw_advanced_features",

    "_safe_div",
    "_clip_series",
    "_rolling_z",
    "_robust_z",
    "_rolling_percentile",
    "_rolling_entropy",
    "_rolling_autocorr",
    "_hurst_approx",
    "_variance_ratio",
    "add_statistical_features",
    "add_raw_statistics_and_calculus",

    "_rolling_slope",
    "_rolling_linear_r2",
    "_rolling_quadratic_curvature",
    "_consecutive_condition_count",
    "add_calculus_features",

    "add_probability_ecosystem_features",
    "add_relative_strength_features",
    "add_price_volume_structure_dots",
    "add_candlestick_features",

    "_squash",
    "add_family_ecosystem_features",
    "add_indicator_internal_dots",
    "add_cross_family_dot_connections",
    "add_final_market_evidence_scores",
    "add_regime_features",

    "build_features_from_df",
    "run_feature_pipeline",
]
'''

(FEATURE_DIR / "__init__.py").write_text(init_content, encoding="utf-8")
print("[OK] Updated features/__init__.py")
