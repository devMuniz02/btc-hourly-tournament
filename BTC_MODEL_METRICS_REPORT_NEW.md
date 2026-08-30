# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T00:43:54.375769+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 119 | 59 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 154 | 94 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 172 | 82 | 90 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 172 | 82 | 90 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| BTC Market Hours | nn | NN | 82 | 47 | 35 | 57.32% | 57.32% | 57.32% | 7.32 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 43 | 39 | 52.44% | 52.44% | 52.44% | 2.44 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Hourly | nn | NN | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| BTC Daily | nn | NN | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 3 | -1.67 |
| BTC Market Hours | transformer | Transformer | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 82 | 34 | 48 | 41.46% | 41.46% | 41.46% | 8.54 pp | -14 | 8 | -1.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 82 | 32 | 50 | 39.02% | 39.02% | 39.02% | 10.98 pp | -18 | 8 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 84 | 32 | 52 | 38.10% | 38.10% | 38.10% | 11.90 pp | -20 | 4 | -5.00 |
| BTC Hourly | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 3 | -5.00 |
| BTC Daily | lstm | LSTM | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 59 | 19 | 40 | 32.20% | 32.20% | 32.20% | 17.80 pp | -21 | 3 | -7.00 |
| BTC Daily | xgb | XGBoost | 94 | 28 | 66 | 29.79% | 29.79% | 29.79% | 20.21 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 3 | 0.33 |
| BTC Hourly | nn | NN | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 3 | -1.67 |
| BTC Hourly | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 3 | -5.00 |
| BTC Hourly | xgb | XGBoost | 59 | 19 | 40 | 32.20% | 32.20% | 32.20% | 17.80 pp | -21 | 3 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 84 | 32 | 52 | 38.10% | 38.10% | 38.10% | 11.90 pp | -20 | 4 | -5.00 |
| BTC Daily | lstm | LSTM | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 94 | 28 | 66 | 29.79% | 29.79% | 29.79% | 20.21 pp | -38 | 5 | -7.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 82 | 47 | 35 | 57.32% | 57.32% | 57.32% | 7.32 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 43 | 39 | 52.44% | 52.44% | 52.44% | 2.44 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 8 | -0.25 |
| BTC Market Hours Daily | rf | RandomForest | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 82 | 34 | 48 | 41.46% | 41.46% | 41.46% | 8.54 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 82 | 32 | 50 | 39.02% | 39.02% | 39.02% | 10.98 pp | -18 | 8 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

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
