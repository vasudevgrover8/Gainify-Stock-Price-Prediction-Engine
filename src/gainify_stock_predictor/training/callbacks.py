"""
Callback helper namespace.

Your original yearly.py creates callbacks directly inside each training stage.
This file is kept for project structure compatibility.
"""


def callbacks_are_defined_inside_stage_functions():
    """
    Returns True because the original callback logic is preserved inside:
    - run_yearly_pretrain
    - run_monthly_finetune
    - run_weekly_finetune
    - run_daily_finetune
    """
    return True
