from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sp500_volatility.metrics import compare
from sp500_volatility.split import time_split
from sp500_volatility.transforms import FEATURE_COLS, TARGET, load_dataset


MODEL_PATH = Path('models') / 'random_forest.joblib'


def make_model(**params) -> RandomForestRegressor:
    defaults = {'n_estimators': 10, 'n_jobs': -1, 'random_state': 42}
    return RandomForestRegressor(**{**defaults, **params})


def train(
    df: pd.DataFrame,
    features: list[str] = FEATURE_COLS,
    target: str = TARGET,
    **params,
    ) -> RandomForestRegressor:
    model = make_model(**params)
    model.fit(df[features], df[target])
    return model


def save_model(model, path: Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def main() -> None:
    df_trn, df_val, _ = time_split(load_dataset())
    model = train(df_trn)
    save_model(model)
    print(f'Saved model to {MODEL_PATH}')
    print('Validation:')
    print(compare(df_val, model.predict(df_val[FEATURE_COLS]), TARGET))


if __name__ == '__main__':
    main()
