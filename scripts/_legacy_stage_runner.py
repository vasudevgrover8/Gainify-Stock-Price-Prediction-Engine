"""
Internal helper for running legacy/yearly.py with a selected RUN_STAGE.

This preserves the original yearly.py logic.
It only creates a temporary runtime copy with RUN_STAGE changed.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_YEARLY = PROJECT_ROOT / "legacy" / "yearly.py"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"


VALID_STAGES = {
    "yearly_pretrain",
    "monthly_finetune",
    "weekly_finetune",
    "daily_finetune",
    "predict_only",
    "full_pipeline",
}


def create_runtime_yearly(stage: str) -> Path:
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage}")

    if not LEGACY_YEARLY.exists():
        raise FileNotFoundError(f"legacy/yearly.py not found at: {LEGACY_YEARLY}")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    source = LEGACY_YEARLY.read_text(encoding="utf-8")

    source = re.sub(
        r'RUN_STAGE\s*=\s*["\'].*?["\']',
        f'RUN_STAGE = "{stage}"',
        source,
        count=1,
    )

    runtime_file = RUNTIME_DIR / f"yearly_runtime_{stage}.py"
    runtime_file.write_text(source, encoding="utf-8")

    return runtime_file


def run_stage(stage: str):
    runtime_file = create_runtime_yearly(stage)

    print(f"[INFO] Running stage: {stage}")
    print(f"[INFO] Runtime file: {runtime_file}")

    result = subprocess.run(
        [sys.executable, str(runtime_file)],
        cwd=str(PROJECT_ROOT),
    )

    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(VALID_STAGES),
        help="Training/prediction stage to run.",
    )

    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
