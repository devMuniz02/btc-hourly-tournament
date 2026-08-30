# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T02:13:01.879322+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 120 | 60 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 156 | 96 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 175 | 84 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 174 | 83 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 0 | 65 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 0 | 65 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 84 | 48 | 36 | 57.14% | 57.14% | 57.14% | 7.14 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Hourly | transformer | Transformer | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 8 | -0.12 |
| BTC Market Hours | rf | RandomForest | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 7 | -0.29 |
| BTC Hourly | nn | NN | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 8 | -1.38 |
| BTC Market Hours | lstm | LSTM | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 8 | -1.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 4 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 8 | -2.38 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| BTC Hourly | rf | RandomForest | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 3 | -3.33 |
| BTC Daily | rf | RandomForest | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 4 | -5.00 |
| BTC Hourly | lstm | LSTM | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 3 | -5.33 |
| BTC Daily | lstm | LSTM | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 60 | 19 | 41 | 31.67% | 31.67% | 31.67% | 18.33 pp | -22 | 3 | -7.33 |
| BTC Daily | xgb | XGBoost | 96 | 29 | 67 | 30.21% | 30.21% | 30.21% | 19.79 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | nn | NN | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 3 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 3 | -2.00 |
| BTC Hourly | rf | RandomForest | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 3 | -5.33 |
| BTC Hourly | xgb | XGBoost | 60 | 19 | 41 | 31.67% | 31.67% | 31.67% | 18.33 pp | -22 | 3 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 4 | -5.00 |
| BTC Daily | lstm | LSTM | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 96 | 29 | 67 | 30.21% | 30.21% | 30.21% | 19.79 pp | -38 | 5 | -7.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 84 | 48 | 36 | 57.14% | 57.14% | 57.14% | 7.14 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | rf | RandomForest | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 8 | -0.12 |
| BTC Market Hours Daily | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 8 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

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
