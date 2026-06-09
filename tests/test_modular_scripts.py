from pathlib import Path


def test_scripts_do_not_use_legacy_stage_runner():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    scripts = [
        "train_yearly.py",
        "train_monthly.py",
        "train_weekly.py",
        "train_daily.py",
        "predict.py",
        "run_full_pipeline.py",
    ]

    for script in scripts:
        text = (scripts_dir / script).read_text(encoding="utf-8")

        assert "_legacy_stage_runner" not in text
        assert "legacy/yearly.py" not in text
        assert "legacy\\yearly.py" not in text
        assert "from gainify_stock_predictor.pipeline" in text


def test_stage_orchestrator_exists():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "gainify_stock_predictor" / "pipeline" / "stage_orchestrator.py"

    text = path.read_text(encoding="utf-8")

    assert "def run_stage" in text
    assert "def build_buckets_for_stage" in text
    assert "run_yearly_pretrain" in text
    assert "run_monthly_finetune" in text
    assert "run_weekly_finetune" in text
    assert "run_daily_finetune" in text
    assert "run_predict_only" in text


def test_pipeline_imports_if_dependencies_available():
    try:
        from gainify_stock_predictor.pipeline.stage_orchestrator import run_stage
    except ImportError as e:
        import pytest
        pytest.skip(f"Pipeline import skipped due missing optional dependency: {e}")

    assert run_stage is not None
