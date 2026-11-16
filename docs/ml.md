# Machine learning

Code: `src/sp500_volatility/train.py`. Exploration: `notebooks/02_split_train_eval.ipynb`.

## Challenges

- **Time dependence:** observations are ordered in time and correlated across stocks on the same day, so a random split would leak the future into training.
- **Size:** about 2.9 million rows, so models must train in reasonable time.
- **Skewed target:** volatility is heavily right-skewed with extreme spikes (2008, 2020).

## Choices

| Choice | Why |
|---|---|
| Time split: train 2000–2018, validation 2019–2021, test 2022 onward (`split.py`) | The model is always evaluated on a period it has never seen |
| Random forest regressor (`train.py`, 10 trees to start) | Handles non-linear interactions between features, needs no scaling, robust to outliers, gives feature importances |

## To do

- TODO: tune the forest on the validation set (number of trees, `min_samples_leaf`, `max_features`).
- TODO: decide whether to predict `log(RV)` instead of `RV`: the notebook argues for it (relative errors, more symmetric target), but the code still predicts `RV` directly.
- TODO: compare with at least one other model (e.g. gradient boosting or a linear model on the same features).
