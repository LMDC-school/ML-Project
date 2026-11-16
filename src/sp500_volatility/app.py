import argparse
from pathlib import Path

from sp500_volatility.metrics import compare
from sp500_volatility.split import time_split
from sp500_volatility.train import MODEL_PATH, load_model
from sp500_volatility.transforms import FEATURE_COLS, PROCESSED_PATH, TARGET, load_dataset


def main(argv: list[str] | None = None) -> None:
    """
    Load the transformed data and the trained model, predict next-week
    volatility on one split and compare the model with both baselines.
    """
    parser = argparse.ArgumentParser(description = 'Predict next-week volatility and compare with the baselines.')
    parser.add_argument('--data', type = Path, default = PROCESSED_PATH)
    parser.add_argument('--model', type = Path, default = MODEL_PATH)
    parser.add_argument('--split', choices = ['train', 'valid', 'test'], default = 'valid')
    args = parser.parse_args(argv)

    splits = dict(zip(['train', 'valid', 'test'], time_split(load_dataset(args.data))))
    df = splits[args.split]
    model = load_model(args.model)

    print(f'{args.split}: {len(df):,} rows, {df["date"].min():%Y-%m-%d} to {df["date"].max():%Y-%m-%d}')
    print(compare(df, model.predict(df[FEATURE_COLS]), TARGET))


if __name__ == '__main__':
    main()
