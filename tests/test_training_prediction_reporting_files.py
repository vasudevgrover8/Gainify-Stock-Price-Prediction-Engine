from pathlib import Path


def test_training_files_exist():
    root = Path(__file__).resolve().parents[1]
    training_dir = root / "src" / "gainify_stock_predictor" / "training"

    required = [
        "yearly_pretrain.py",
        "monthly_finetune.py",
        "weekly_finetune.py",
        "daily_finetune.py",
        "trainer_utils.py",
        "losses.py",
        "callbacks.py",
        "__init__.py",
    ]

    for file_name in required:
        assert (training_dir / file_name).exists(), f"Missing training file: {file_name}"


def test_checkpoint_files_exist():
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "src" / "gainify_stock_predictor" / "checkpoints"

    required = [
        "checkpoint_manager.py",
        "metadata_manager.py",
        "__init__.py",
    ]

    for file_name in required:
        assert (checkpoint_dir / file_name).exists(), f"Missing checkpoint file: {file_name}"


def test_prediction_reporting_files_exist():
    root = Path(__file__).resolve().parents[1]

    prediction_dir = root / "src" / "gainify_stock_predictor" / "prediction"
    reporting_dir = root / "src" / "gainify_stock_predictor" / "reporting"

    prediction_required = [
        "predictor.py",
        "signal_builder.py",
        "ranker.py",
        "__init__.py",
    ]

    reporting_required = [
        "prediction_writer.py",
        "ranked_reports.py",
        "metrics.py",
        "__init__.py",
    ]

    for file_name in prediction_required:
        assert (prediction_dir / file_name).exists(), f"Missing prediction file: {file_name}"

    for file_name in reporting_required:
        assert (reporting_dir / file_name).exists(), f"Missing reporting file: {file_name}"


def test_training_functions_extracted():
    root = Path(__file__).resolve().parents[1]
    training_dir = root / "src" / "gainify_stock_predictor" / "training"

    assert "def pretrain_bucket" in (training_dir / "yearly_pretrain.py").read_text(encoding="utf-8")
    assert "def run_yearly_pretrain" in (training_dir / "yearly_pretrain.py").read_text(encoding="utf-8")
    assert "def run_monthly_finetune" in (training_dir / "monthly_finetune.py").read_text(encoding="utf-8")
    assert "def run_weekly_finetune" in (training_dir / "weekly_finetune.py").read_text(encoding="utf-8")
    assert "def run_daily_finetune" in (training_dir / "daily_finetune.py").read_text(encoding="utf-8")

    losses_text = (training_dir / "losses.py").read_text(encoding="utf-8")
    assert "def spike_weighted_mse" in losses_text
    assert "def smooth_huber" in losses_text
    assert "def focal_bce_soft" in losses_text
    assert "LOSSES" in losses_text


def test_checkpoint_prediction_reporting_functions_extracted():
    root = Path(__file__).resolve().parents[1]

    checkpoint_dir = root / "src" / "gainify_stock_predictor" / "checkpoints"
    prediction_dir = root / "src" / "gainify_stock_predictor" / "prediction"
    reporting_dir = root / "src" / "gainify_stock_predictor" / "reporting"

    checkpoint_text = (checkpoint_dir / "checkpoint_manager.py").read_text(encoding="utf-8")
    metadata_text = (checkpoint_dir / "metadata_manager.py").read_text(encoding="utf-8")
    predictor_text = (prediction_dir / "predictor.py").read_text(encoding="utf-8")
    signal_text = (prediction_dir / "signal_builder.py").read_text(encoding="utf-8")
    writer_text = (reporting_dir / "prediction_writer.py").read_text(encoding="utf-8")
    ranked_text = (reporting_dir / "ranked_reports.py").read_text(encoding="utf-8")
    metrics_text = (reporting_dir / "metrics.py").read_text(encoding="utf-8")

    assert "def get_stage_output_dir" in checkpoint_text
    assert "def load_latest_successful_checkpoint" in checkpoint_text
    assert "def resolve_parent_checkpoint" in checkpoint_text
    assert "def save_stage_metadata" in metadata_text

    assert "def forecast_1d" in predictor_text
    assert "def run_predict_only" in predictor_text
    assert "def calculate_verdict" in signal_text
    assert "def ensemble_next_day_signal" in signal_text

    assert "def save_master_predictions_csv" in writer_text
    assert "def build_ranked_volatility_reports" in ranked_text
    assert "def evaluate_holdout_close" in metrics_text
