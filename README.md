# Predicting Next-Week Stock Volatility on the S&P 500

End-of-course Machine Learning project.

## Goal

The goal of this project is to **predict how much each S&P 500 stock will move over the next 5 trading days** (its realized volatility), using only information available at the end of the current week.

Predicting the *direction* of stock prices is close to impossible: price-prediction models usually just learn that "tomorrow looks like today." Volatility is different. It tends to cluster, with calm periods following calm periods and turbulent periods following turbulent ones, which makes it genuinely predictable.

The central question of the project is:

> **Can machine learning models predict next-week volatility better than the simple rule "next week will be as volatile as this week," and better than the market's own forecast (the VIX)?**

This is a supervised regression problem on panel data (many stocks observed over many weeks).

## Data sources

### 1. S&P 500 Stocks — Kaggle
[kaggle.com/datasets/darkmatternet/sp-500-stocks](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily) — License: CC0 (public domain)

| File | Content | Use in this project |
|---|---|---|
| `sp500_stocks.csv` | Daily open, high, low, close, adjusted close and volume for each stock | Main data: all stock-level features and the target |
| `sp500_companies.csv` | One row per company: sector, industry, market cap, etc. | Only the `Sector` column |

### 2. FRED — Federal Reserve Bank of St. Louis
Free economic data, downloaded directly by the script.

| Series | Meaning | Source |
|---|---|---|
| `VIXCLS` | VIX: the volatility the market expects over the next 30 days | https://fred.stlouisfed.org/series/VIXCLS |
| `DGS10` | 10-year US Treasury interest rate | https://fred.stlouisfed.org/series/DGS10 |
| `T10Y2Y` | Difference between 10-year and 2-year rates (a classic recession signal) | https://fred.stlouisfed.org/series/T10Y2Y |

### Avoiding data leakage

- Every feature uses only data available at the end of the week; the target uses only the following 5 trading days.
- Economic indicators are shifted by one day so the model never sees information published later.
- Company figures such as market cap are **not** used, because the file contains today's values, not historical ones.
- The data is split **by time**, never randomly, so the model is always tested on a period it has not seen.

## Limitations

- **Survivorship bias:** the Kaggle dataset only includes companies that are in the S&P 500 today. Companies that failed or left the index are missing.
- **Sector is a current snapshot:** a few companies may have changed sector over time.
- **Market proxy:** market volatility is computed as the equal-weighted average of all stocks, not from the official index.
- The validation period includes the **COVID-19 crash of 2020**, an extreme event that makes it a demanding test.

