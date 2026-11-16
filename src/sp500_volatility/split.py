import pandas as pd


# Split by time, never randomly, so the model is always tested on a period it has not seen
TRAIN_END = '2018-12-31'
VALID_END = '2021-12-31'


def time_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
    date_col: str = 'date',
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train (up to train_end), validation (up to valid_end)
    and test (everything after).
    """
    dates = df[date_col]
    train = df[dates <= train_end]
    valid = df[(dates > train_end) & (dates <= valid_end)]
    test = df[dates > valid_end]
    return train.copy(), valid.copy(), test.copy()


def split_vals(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[:n].copy(), df[n:].copy()
