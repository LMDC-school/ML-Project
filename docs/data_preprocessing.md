# Data preprocessing

Code: `src/sp500_volatility/transforms.py` (`build_dataset`). Exploration: `notebooks/01_transforms.ipynb`.

## Pipeline

1. **Load** the stock prices, company table and the three FRED series (`load_raw`).
2. **Merge** company info on `symbol` and the FRED series on `date` (`merge_datasets`).
3. **Fill** missing macro values (`fill_macro`).
4. **Compute the target** (`add_target`) and **the features** (`add_features`).
5. **Drop** rows where a feature or the target can't be computed (`drop_incomplete`).
6. **Save** to `data/processed/dataset.parquet` (`uv run build-dataset`).

## Challenge 1: missing values

The merged dataset has 43,164 missing values (about 0.1% of cells), all in the macro columns:

| Column | Missing | Cause |
|---|---|---|
| `DGS10` | 21,582 | Bond-market holidays (Columbus Day, Veterans Day): stocks trade, Treasuries don't |
| `T10Y2Y` | 21,079 | Same holidays |
| `VIXCLS` | 503 | Days when the VIX was not published |
| Prices and categories | 0 | Complete |

**Choice:** forward-fill by date. The gaps are whole dates (every stock misses the value that day), and on a holiday the best estimate of a rate is the previous day's rate. We fill a one-row-per-date table and merge it back, rather than filling per stock.

## Challenge 2: defining the target

**Choice:** realized volatility over the next 5 trading days, per stock:

$$RV_{t \to t+5} = \sqrt{\sum_{k=1}^{5} r_{t+k}^2}$$

where $r$ is the daily log return. Five days matches the "next week" horizon of the question.

## Challenge 3: features without leakage

Every feature uses only past data and is computed per stock (`groupby('symbol')`), so the future never leaks into the present and one stock never leaks into another.

| Feature | Meaning | Why it should help |
|---|---|---|
| `rv_5d`, `rv_20d`, `rv_60d` | Past realized volatility over 1 week / 1 month / 3 months, rescaled to a weekly horizon | Volatility clusters |
| `range_5d` | Average daily high-low range over the week | Intraday turbulence that closing prices miss |
| `ret`, `ret_sq` | Today's return and its square | Most recent shock |
| `ret_5d`, `ret_20d` | Past weekly / monthly return | Drops are followed by higher volatility (leverage effect) |
| `volume_z` | This week's volume vs its 3-month average | Unusual activity comes with news |

**Not used:** company figures such as market cap, because the file holds today's values, not historical ones.

## Open points

- TODO: the README promised that macro indicators are shifted by one day (published values may arrive after the close). `fill_macro` does not shift yet.
- TODO: the macro columns and `sector` are in the dataset but not yet in `FEATURE_COLS`.
