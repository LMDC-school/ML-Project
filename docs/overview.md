# Overview

## Objective

Predict **how much each S&P 500 stock will move over the next 5 trading days**: its realized volatility, using only information available at the end of the current day.

This is a supervised regression problem on panel data (about 500 stocks observed daily since 2000).

## Expected added value

Predicting the *direction* of prices is close to impossible: models end up learning that "tomorrow looks like today." Volatility is different. It clusters (calm periods follow calm periods, turbulent ones follow turbulent ones), which makes it genuinely predictable, and it is what matters for risk management, position sizing and option pricing.

The central question:

> **Can a machine learning model predict next-week volatility better than the rule "next week will be as volatile as this week," and better than the market's own forecast (the VIX)?**

A model only adds value if it beats both baselines on a period it has never seen.

## Data

| Source | Files | Use |
|---|---|---|
| [Kaggle: S&P 500 stocks](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily) (CC0) | `sp500_stocks.csv.xz`, `sp500_companies.csv` | Daily prices and volumes (features and target); company sector |
| [FRED](https://fred.stlouisfed.org/) | `VIXCLS.csv`, `DGS10.csv`, `T10Y2Y.csv` | VIX, 10-year Treasury rate, 10Y-2Y spread |

After merging: 2,949,733 rows (one per stock and trading day).

## Limitations

- **Survivorship bias:** the Kaggle dataset only contains today's S&P 500 members; companies that failed or left the index are missing.
- **Sector is a current snapshot:** a few companies may have changed sector over time.
- **Market proxy:** the VIX measures the S&P 500 index, not individual stocks, so as a stock-level forecast it is only a rough benchmark.
- The validation period contains the **COVID-19 crash of 2020**, an extreme event that makes it a demanding test.
