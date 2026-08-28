# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T21:46:05.694476+00:00
Scope: `new`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 38 | 64 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 134 | 74 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 136 | 62 | 74 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 135 | 61 | 74 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 0 | 47 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 0 | 47 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 62 | 35 | 27 | 56.45% | 56.45% | 56.45% | 6.45 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | transformer | Transformer | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| BTC Hourly | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 6 | -2.17 |
| BTC Market Hours | lstm | LSTM | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 61 | 20 | 41 | 32.79% | 32.79% | 32.79% | 17.21 pp | -21 | 6 | -3.50 |
| BTC Daily | rf | RandomForest | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 3 | -5.33 |
| BTC Daily | lstm | LSTM | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 74 | 25 | 49 | 33.78% | 33.78% | 33.78% | 16.22 pp | -24 | 4 | -6.00 |
| BTC Hourly | rf | RandomForest | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 2 | -7.00 |
| BTC Hourly | xgb | XGBoost | 38 | 11 | 27 | 28.95% | 28.95% | 28.95% | 21.05 pp | -16 | 2 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 2 | -7.00 |
| BTC Hourly | xgb | XGBoost | 38 | 11 | 27 | 28.95% | 28.95% | 28.95% | 21.05 pp | -16 | 2 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| BTC Daily | rf | RandomForest | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 3 | -5.33 |
| BTC Daily | lstm | LSTM | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 74 | 25 | 49 | 33.78% | 33.78% | 33.78% | 16.22 pp | -24 | 4 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 62 | 35 | 27 | 56.45% | 56.45% | 56.45% | 6.45 pp | 8 | 5 | 1.60 |
| BTC Market Hours | rf | RandomForest | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | transformer | Transformer | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 61 | 20 | 41 | 32.79% | 32.79% | 32.79% | 17.21 pp | -21 | 6 | -3.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
