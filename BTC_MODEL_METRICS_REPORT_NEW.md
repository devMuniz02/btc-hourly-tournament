# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T06:19:28.362667+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 174 | 114 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 209 | 149 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 267 | 137 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 267 | 137 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 137 | 72 | 65 | 52.55% | 52.55% | 52.55% | 2.55 pp | 7 | 11 | 0.64 |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 12 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | rf | RandomForest | 137 | 65 | 72 | 47.45% | 47.45% | 47.45% | 2.55 pp | -7 | 11 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 139 | 67 | 72 | 48.20% | 48.20% | 48.20% | 1.80 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 12 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 12 | -1.25 |
| BTC Market Hours | transformer | Transformer | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Daily | transformer | Transformer | 139 | 60 | 79 | 43.17% | 43.17% | 43.17% | 6.83 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 137 | 52 | 85 | 37.96% | 37.96% | 37.96% | 12.04 pp | -33 | 12 | -2.75 |
| BTC Market Hours | lstm | LSTM | 137 | 52 | 85 | 37.96% | 37.96% | 37.96% | 12.04 pp | -33 | 11 | -3.00 |
| BTC Daily | rf | RandomForest | 139 | 57 | 82 | 41.01% | 41.01% | 41.01% | 8.99 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 5 | -3.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 149 | 53 | 96 | 35.57% | 35.57% | 35.57% | 14.43 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 139 | 49 | 90 | 35.25% | 35.25% | 35.25% | 14.75 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 114 | 41 | 73 | 35.96% | 35.96% | 35.96% | 14.04 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 114 | 37 | 77 | 32.46% | 32.46% | 32.46% | 17.54 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 114 | 41 | 73 | 35.96% | 35.96% | 35.96% | 14.04 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 114 | 37 | 77 | 32.46% | 32.46% | 32.46% | 17.54 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 139 | 67 | 72 | 48.20% | 48.20% | 48.20% | 1.80 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 139 | 60 | 79 | 43.17% | 43.17% | 43.17% | 6.83 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 139 | 57 | 82 | 41.01% | 41.01% | 41.01% | 8.99 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 149 | 53 | 96 | 35.57% | 35.57% | 35.57% | 14.43 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 139 | 49 | 90 | 35.25% | 35.25% | 35.25% | 14.75 pp | -41 | 7 | -5.86 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 137 | 72 | 65 | 52.55% | 52.55% | 52.55% | 2.55 pp | 7 | 11 | 0.64 |
| BTC Market Hours | rf | RandomForest | 137 | 65 | 72 | 47.45% | 47.45% | 47.45% | 2.55 pp | -7 | 11 | -0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| BTC Market Hours | transformer | Transformer | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 137 | 52 | 85 | 37.96% | 37.96% | 37.96% | 12.04 pp | -33 | 11 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 12 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 137 | 52 | 85 | 37.96% | 37.96% | 37.96% | 12.04 pp | -33 | 12 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
