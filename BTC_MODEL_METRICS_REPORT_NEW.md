# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T00:10:43.263918+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 118 | 58 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 154 | 94 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 172 | 82 | 90 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 172 | 82 | 90 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 15:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 15:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 15:00:00+00:00 | 65 | 1 | 64 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 15:00:00+00:00 | 65 | 1 | 64 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 82 | 47 | 35 | 57.32% | 57.32% | 57.32% | 7.32 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 43 | 39 | 52.44% | 52.44% | 52.44% | 2.44 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 7 | 0.43 |
| BTC Hourly | nn | NN | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 7 | -0.29 |
| BTC Daily | transformer | Transformer | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 8 | -1.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 84 | 39 | 45 | 46.43% | 46.43% | 46.43% | 3.57 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 82 | 34 | 48 | 41.46% | 41.46% | 41.46% | 8.54 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 82 | 32 | 50 | 39.02% | 39.02% | 39.02% | 10.98 pp | -18 | 8 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| BTC Hourly | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Daily | rf | RandomForest | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 4 | -4.50 |
| BTC Hourly | lstm | LSTM | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 58 | 19 | 39 | 32.76% | 32.76% | 32.76% | 17.24 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 94 | 28 | 66 | 29.79% | 29.79% | 29.79% | 20.21 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 3 | -4.67 |
| BTC Hourly | xgb | XGBoost | 58 | 19 | 39 | 32.76% | 32.76% | 32.76% | 17.24 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 84 | 40 | 44 | 47.62% | 47.62% | 47.62% | 2.38 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 84 | 39 | 45 | 46.43% | 46.43% | 46.43% | 3.57 pp | -6 | 4 | -1.50 |
| BTC Daily | rf | RandomForest | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 4 | -4.50 |
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
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | lstm | LSTM | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
