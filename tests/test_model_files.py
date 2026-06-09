from pathlib import Path


def test_model_files_exist():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "src" / "gainify_stock_predictor" / "models"

    required_files = [
        "layers.py",
        "cnn_encoder.py",
        "tft_blocks.py",
        "attention_blocks.py",
        "heads.py",
        "main_model.py",
        "tree_models.py",
        "__init__.py",
    ]

    for file_name in required_files:
        assert (model_dir / file_name).exists(), f"Missing: {file_name}"


def test_model_functions_were_extracted():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "src" / "gainify_stock_predictor" / "models"

    layers_text = (model_dir / "layers.py").read_text(encoding="utf-8")
    cnn_text = (model_dir / "cnn_encoder.py").read_text(encoding="utf-8")
    heads_text = (model_dir / "heads.py").read_text(encoding="utf-8")
    main_text = (model_dir / "main_model.py").read_text(encoding="utf-8")
    tree_text = (model_dir / "tree_models.py").read_text(encoding="utf-8")

    assert "class GatingLayer" in layers_text
    assert "class PositionalEncoding" in layers_text

    assert "def build_advanced_encoder" in cnn_text

    assert "def add_adapter" in heads_text
    assert "def _build_head" in heads_text

    assert "def build_pretrain_model" in main_text
    assert "def build_multitask_model" in main_text

    assert "def build_tabular_dataset_1d" in tree_text
    assert "def train_lgbm_1d" in tree_text
    assert "def train_xgb_1d" in tree_text
