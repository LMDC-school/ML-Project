import numpy as np
import pandas as pd
from sklearn import metrics


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """
    Score predictions of next-week realized volatility.
    """
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': float(metrics.mean_absolute_error(y_true, y_pred)),
        'r2': float(metrics.r2_score(y_true, y_pred)),
    }


def persistence_baseline(df: pd.DataFrame) -> pd.Series:
    """
    "Next week will be as volatile as this week."
    """
    return df['rv_5d']


def vix_baseline(df: pd.DataFrame) -> pd.Series:
    """
    The market's own forecast: the VIX (annualized %, S&P 500 index level)
    rescaled to a 5-trading-day horizon.
    """
    return df['VIXCLS'] / 100 * np.sqrt(5 / 252)


def compare(df: pd.DataFrame, y_pred, target: str) -> pd.DataFrame:
    """
    One row per predictor (model and both baselines), one column per metric.
    """
    y_true = df[target]
    scores = {
        'model': evaluate(y_true, y_pred),
        'persistence': evaluate(y_true, persistence_baseline(df)),
        'vix': evaluate(y_true, vix_baseline(df)),
    }
    return pd.DataFrame(scores).T
