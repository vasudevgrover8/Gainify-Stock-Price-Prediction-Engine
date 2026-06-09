"""
Scaling utilities.

These are safe wrappers only.
They do not change the original scaler logic inside yearly.py.
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def create_minmax_scaler():
    """
    Return sklearn MinMaxScaler.
    """
    return MinMaxScaler()


def create_standard_scaler():
    """
    Return sklearn StandardScaler.
    """
    return StandardScaler()