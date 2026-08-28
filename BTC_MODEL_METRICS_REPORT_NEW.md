# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T08:50:07.813694+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 28 | 74 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 124 | 64 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 117 | 52 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 116 | 51 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| BTC Market Hours | nn | NN | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | nn | NN | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 5 | -0.20 |
| BTC Market Hours | rf | RandomForest | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Hourly | lstm | LSTM | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | nn | NN | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 3 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Market Hours | lstm | LSTM | 52 | 18 | 34 | 34.62% | 34.62% | 34.62% | 15.38 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 51 | 15 | 36 | 29.41% | 29.41% | 29.41% | 20.59 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 54 | 18 | 36 | 33.33% | 33.33% | 33.33% | 16.67 pp | -18 | 3 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 2 | 3.00 |
| BTC Hourly | nn | NN | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | rf | RandomForest | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 54 | 18 | 36 | 33.33% | 33.33% | 33.33% | 16.67 pp | -18 | 3 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 4 | 1.50 |
| BTC Market Hours | rf | RandomForest | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| BTC Market Hours | lstm | LSTM | 52 | 18 | 34 | 34.62% | 34.62% | 34.62% | 15.38 pp | -16 | 4 | -4.00 |

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
