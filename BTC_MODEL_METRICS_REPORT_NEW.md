# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T04:19:53.178887+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 156 | 96 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 192 | 132 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 237 | 120 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 237 | 120 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 100 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 120 | 64 | 56 | 53.33% | 53.33% | 53.33% | 3.33 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | rf | RandomForest | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 10 | -0.20 |
| BTC Hourly | nn | NN | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 120 | 56 | 64 | 46.67% | 46.67% | 46.67% | 3.33 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 96 | 45 | 51 | 46.88% | 46.88% | 46.88% | 3.12 pp | -6 | 4 | -1.50 |
| BTC Daily | nn | NN | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 120 | 51 | 69 | 42.50% | 42.50% | 42.50% | 7.50 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 120 | 49 | 71 | 40.83% | 40.83% | 40.83% | 9.17 pp | -22 | 10 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 11 | -2.36 |
| BTC Market Hours | lstm | LSTM | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 11 | -2.73 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 122 | 51 | 71 | 41.80% | 41.80% | 41.80% | 8.20 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 96 | 39 | 57 | 40.62% | 40.62% | 40.62% | 9.38 pp | -18 | 4 | -4.50 |
| BTC Daily | xgb | XGBoost | 132 | 48 | 84 | 36.36% | 36.36% | 36.36% | 13.64 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 122 | 42 | 80 | 34.43% | 34.43% | 34.43% | 15.57 pp | -38 | 6 | -6.33 |
| BTC Hourly | xgb | XGBoost | 96 | 32 | 64 | 33.33% | 33.33% | 33.33% | 16.67 pp | -32 | 4 | -8.00 |
| BTC Hourly | lstm | LSTM | 96 | 30 | 66 | 31.25% | 31.25% | 31.25% | 18.75 pp | -36 | 4 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 96 | 45 | 51 | 46.88% | 46.88% | 46.88% | 3.12 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 96 | 39 | 57 | 40.62% | 40.62% | 40.62% | 9.38 pp | -18 | 4 | -4.50 |
| BTC Hourly | xgb | XGBoost | 96 | 32 | 64 | 33.33% | 33.33% | 33.33% | 16.67 pp | -32 | 4 | -8.00 |
| BTC Hourly | lstm | LSTM | 96 | 30 | 66 | 31.25% | 31.25% | 31.25% | 18.75 pp | -36 | 4 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | nn | NN | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 6 | -1.67 |
| BTC Daily | rf | RandomForest | 122 | 51 | 71 | 41.80% | 41.80% | 41.80% | 8.20 pp | -20 | 6 | -3.33 |
| BTC Daily | xgb | XGBoost | 132 | 48 | 84 | 36.36% | 36.36% | 36.36% | 13.64 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 122 | 42 | 80 | 34.43% | 34.43% | 34.43% | 15.57 pp | -38 | 6 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 120 | 64 | 56 | 53.33% | 53.33% | 53.33% | 3.33 pp | 8 | 10 | 0.80 |
| BTC Market Hours | rf | RandomForest | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 10 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 120 | 56 | 64 | 46.67% | 46.67% | 46.67% | 3.33 pp | -8 | 10 | -0.80 |
| BTC Market Hours | transformer | Transformer | 120 | 51 | 69 | 42.50% | 42.50% | 42.50% | 7.50 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 120 | 49 | 71 | 40.83% | 40.83% | 40.83% | 9.17 pp | -22 | 10 | -2.20 |
| BTC Market Hours | lstm | LSTM | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 10 | -2.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | nn | NN | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | xgb | XGBoost | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 11 | -2.73 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
