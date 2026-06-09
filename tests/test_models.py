import numpy as np
import pandas as pd
import pytest


tf = pytest.importorskip("tensorflow")


def test_model_module_imports():
    import gainify_stock_predictor.models as models

    assert models.GatingLayer is not None
    assert models.PositionalEncoding is not None
    assert models.build_advanced_encoder is not None
    assert models.build_pretrain_model is not None
    assert models.build_multitask_model is not None
    assert models.build_tabular_dataset_1d is not None


def test_custom_layers_exist():
    from gainify_stock_predictor.models import GatingLayer, PositionalEncoding

    assert GatingLayer is not None
    assert PositionalEncoding is not None


def test_build_multitask_model_small_shape():
    from gainify_stock_predictor.models import build_multitask_model

    model = build_multitask_model(seq_len=10, n_features=5)

    assert model is not None
    assert len(model.inputs) == 3
    assert len(model.outputs) == 3


def test_build_tabular_dataset_1d():
    from gainify_stock_predictor.models import build_tabular_dataset_1d

    df = pd.DataFrame({
        "f1": np.arange(20, dtype=float),
        "f2": np.arange(20, dtype=float) * 2,
        "LogRet": np.linspace(-0.01, 0.01, 20),
    })

    X, y = build_tabular_dataset_1d(df, ["f1", "f2"])

    assert X.shape[1] == 2
    assert len(X) == len(y)
