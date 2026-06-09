from pathlib import Path
import py_compile


def test_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    required = [
        "_legacy_stage_runner.py",
        "fetch_data.py",
        "update_data.py",
        "train_yearly.py",
        "train_monthly.py",
        "train_weekly.py",
        "train_daily.py",
        "predict.py",
        "run_full_pipeline.py",
    ]

    for file_name in required:
        assert (scripts_dir / file_name).exists(), f"Missing script: {file_name}"


def test_scripts_compile():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    for path in scripts_dir.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
