# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T08:29:54.700466+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 27 | 75 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 123 | 63 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 116 | 51 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 116 | 51 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 2 | 3.50 |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| BTC Market Hours | nn | NN | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| BTC Hourly | nn | NN | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 5 | -0.20 |
| BTC Market Hours | rf | RandomForest | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Hourly | lstm | LSTM | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | nn | NN | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 3 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| BTC Hourly | rf | RandomForest | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |
| BTC Hourly | xgb | XGBoost | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| BTC Market Hours Daily | lstm | LSTM | 51 | 15 | 36 | 29.41% | 29.41% | 29.41% | 20.59 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 4 | -4.25 |
| BTC Market Hours | lstm | LSTM | 51 | 17 | 34 | 33.33% | 33.33% | 33.33% | 16.67 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 53 | 18 | 35 | 33.96% | 33.96% | 33.96% | 16.04 pp | -17 | 3 | -5.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 2 | 3.50 |
| BTC Hourly | nn | NN | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Hourly | lstm | LSTM | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| BTC Hourly | rf | RandomForest | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |
| BTC Hourly | xgb | XGBoost | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 3 | 0.33 |
| BTC Daily | nn | NN | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 3 | -2.33 |
| BTC Daily | xgb | XGBoost | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 53 | 18 | 35 | 33.96% | 33.96% | 33.96% | 16.04 pp | -17 | 3 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 4 | 1.25 |
| BTC Market Hours | rf | RandomForest | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| BTC Market Hours | lstm | LSTM | 51 | 17 | 34 | 33.33% | 33.33% | 33.33% | 16.67 pp | -17 | 4 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 51 | 15 | 36 | 29.41% | 29.41% | 29.41% | 20.59 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

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
