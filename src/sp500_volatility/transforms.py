from pathlib import Path
from typing import Callable
import copy
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype


DATA_DIR = Path('data')
PROCESSED_PATH = DATA_DIR / 'processed' / 'dataset.parquet'

MACRO_COLS = ['VIXCLS', 'DGS10', 'T10Y2Y']
TARGET = 'rv_next5'
FEATURE_COLS = [
    'ret', 'ret_sq', 'rv_5d', 'rv_20d', 'rv_60d',
    'range_5d', 'ret_5d', 'ret_20d', 'volume_z',
]


# ---------------------------------------------------------------------------
# Project pipeline: raw files -> merged -> cleaned -> target + features
# ---------------------------------------------------------------------------

def load_raw(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Load the Kaggle S&P 500 files and the three FRED series.
    """
    data_dir = Path(data_dir)
    raw = {
        'stocks': pd.read_csv(data_dir / 'sp500_stocks.csv.xz', parse_dates = ['date']),
        'companies': pd.read_csv(data_dir / 'sp500_companies.csv', parse_dates = ['date_added']),
    }
    for col in MACRO_COLS:
        raw[col] = pd.read_csv(data_dir / f'{col}.csv', parse_dates = ['observation_date'])
    return raw


def merge_datasets(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join company info (on symbol) and the FRED series (on date) to the daily prices.
    """
    comp = raw['companies'][['symbol', 'sector', 'company', 'sub_industry']]
    df = raw['stocks'].merge(comp, on = 'symbol', how = 'left')

    for col in MACRO_COLS:
        fred = raw[col][['observation_date', col]]
        df = df.merge(fred, left_on = 'date', right_on = 'observation_date', how = 'left')
        df = df.drop(columns = 'observation_date')
    return df


def fill_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill the macro indicators by date: on a bond-market holiday,
    use the previous trading day's value.
    """
    macro = (df.groupby('date')[MACRO_COLS].first()
               .sort_index()
               .ffill())

    df = df.drop(columns = MACRO_COLS).merge(macro, on = 'date', how = 'left')
    return df.sort_values(['symbol', 'date']).reset_index(drop = True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily log returns and the target: realized volatility over the next
    5 trading days, sqrt(sum of the next 5 squared log returns), per stock.
    """
    df = df.sort_values(['symbol', 'date']).reset_index(drop = True)
    df['log_close'] = np.log(df['close'])
    df['ret'] = df.groupby('symbol')['log_close'].diff()
    df['ret_sq'] = df['ret'] ** 2

    g = df.groupby('symbol')
    df[TARGET] = np.sqrt(sum(g['ret_sq'].shift(-k) for k in range(1, 6)))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the stock-level features. Each one uses only past data and is computed
    per stock, so one stock never leaks into another. Expects add_target first.
    """
    g = df.groupby('symbol')

    def past_rv(k):
        # Rolling realized volatility over k days, rescaled to a weekly horizon
        return np.sqrt(g['ret_sq'].transform(lambda s: s.rolling(k).mean()) * 5)

    df['rv_5d'] = past_rv(5)
    df['rv_20d'] = past_rv(20)
    df['rv_60d'] = past_rv(60)

    df['hl_range'] = np.log(df['high'] / df['low'])
    df['range_5d'] = df.groupby('symbol')['hl_range'].transform(lambda s: s.rolling(5).mean())

    df['ret_5d'] = g['log_close'].diff(5)
    df['ret_20d'] = g['log_close'].diff(20)

    df['log_volume'] = np.log1p(df['volume'])
    g = df.groupby('symbol')
    df['volume_z'] = (g['log_volume'].transform(lambda s: s.rolling(5).mean())
                      - g['log_volume'].transform(lambda s: s.rolling(60).mean()))
    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where a feature or the target can't be computed
    (start and end of each stock's history, zero prices or volumes).
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset = FEATURE_COLS + [TARGET]).reset_index(drop = True)


def build_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Run the full pipeline, from the raw files to the model-ready dataset.
    """
    df = merge_datasets(load_raw(data_dir))
    df = fill_macro(df)
    df = add_target(df)
    df = add_features(df)
    return drop_incomplete(df)


def save_dataset(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(path, index = False)


def load_dataset(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Generic helpers from the course
# ---------------------------------------------------------------------------

def add_date_columns(df: pd.DataFrame, fields: list[str], drop: bool = True) -> pd.DataFrame:
    """
    Convert a column of df from a datetime64 to many columns containing
    the information from the date.
    """
    df = copy.deepcopy(df)
        
    for field in fields:
        fld = df[field]
        fld_dtype = fld.dtype
        
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[field] = fld = pd.to_datetime(fld, infer_datetime_format = True)
            
        targ_pre = re.sub('[Dd]ate$', '', field)
        attr = [
            'Year', 'Month', 'Day', 
            'Dayofweek', 'Dayofyear',
            'Is_month_start', 'Is_month_end',
            'Is_quarter_start', 'Is_quarter_end',
            'Is_year_start', 'Is_year_end',
        ]
        for n in attr: 
            df[targ_pre + n] = getattr(fld.dt, n.lower())
            
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        
    if drop: 
        df.drop(columns = fields, axis = 1, inplace = True)
    return df

    
def is_date(x: pd.Series) -> bool: 
    """
    Assert whether a pandas Series is of dtype np.datetime64.
    """
    return np.issubdtype(x.dtype, np.datetime64)


def change_columns_from_str_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change any columns of strings in a panda's dataframe to a column of
    categorical values.
    """
    df = copy.deepcopy(df)
    
    for n, c in df.items():
        if is_string_dtype(c): 
            df[n] = c.astype('category').cat.as_ordered()
    return df

    
def apply_cats(df, trn):
    """
    Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.
    """
    for n, c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name == 'category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(trn[n].cat.categories, ordered = True, inplace = True)
    return


def fix_missing(df, col, name, na_dict):
    """
    Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

    
def numericalize(df: pd.DataFrame, col: str, name: str, max_n_cat: int | None) -> pd.DataFrame:
    """
    Changes the column col from a categorical type to it's integer codes.
    """
    df = copy.deepcopy(df)
    if (not is_numeric_dtype(col) 
        and (max_n_cat is None or len(col.cat.categories) > max_n_cat)):
        df[name] = pd.Categorical(col).codes + 1
    return df


def process_df(
    df: pd.DataFrame, 
    y_field: str | None = None, 
    skip_flds: list = [],
    ignore_flds: list = [], 
    na_dict: dict = {},
    preproc_fn: Callable = None, 
    max_n_cat = None, 
    ):
    """
    Take a dataframe df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df 
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.
    """
    df = copy.deepcopy(df)
    
    df_ignored = df.loc[:, ignore_flds]
    df = df.drop(columns = ignore_flds)
    
    if preproc_fn: 
        preproc_fn(df)
        
    if y_field is None: 
        y = None
    else:
        if not is_numeric_dtype(df[y_field]): 
            df[y_field] = pd.Categorical(df[y_field]).codes
        y = df[y_field].values
        skip_flds += [y_field]
    
    df = df.drop(columns = skip_flds)

    na_dict_initial = na_dict.copy()
    for n, c in df.items(): 
        na_dict = fix_missing(df, c, n, na_dict)
    
    if len(na_dict_initial) > 0:
        df = df.drop(
            [a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], 
            axis = 1,
        )
    for n,c in df.items(): 
        df = numericalize(df, c, n, max_n_cat)
        
    df = pd.get_dummies(df, dummy_na = True)
    df = pd.concat([df_ignored, df], axis = 1)
    return (df, y, na_dict)


def main() -> None:
    dataset = build_dataset()
    save_dataset(dataset)
    print(f'Saved {len(dataset):,} rows x {dataset.shape[1]} columns to {PROCESSED_PATH}')


if __name__ == '__main__':
    main()