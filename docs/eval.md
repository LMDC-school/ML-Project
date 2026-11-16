# Evaluation

Code: `src/sp500_volatility/metrics.py`, `src/sp500_volatility/app.py`.

## Protocol

1. Tune on the **validation** set (2019–2021, includes the COVID crash).
2. Retrain on train + validation with the chosen settings.
3. Report once on the **test** set (2022 onward), never used before.

Every predictor is scored with RMSE, MAE and R² on next-week realized volatility, against two baselines:

| Baseline | Prediction |
|---|---|
| Persistence | Next week's volatility = this week's (`rv_5d`) |
| VIX | The VIX rescaled to 5 trading days: `VIXCLS / 100 * sqrt(5 / 252)` |

Run with `uv run app --split valid` (or `--split test`).

## Results

TODO: fill in once the model is trained on the full dataset.

| Predictor | RMSE | MAE | R² |
|---|---|---|---|
| Random forest | | | |
| Persistence | | | |
| VIX | | | |

## Interpretation

TODO.

## Conclusion

TODO: answer the central question: does the model beat both baselines on the test period?
