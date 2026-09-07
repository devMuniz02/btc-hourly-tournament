# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T01:45:27.096633+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 252 | 192 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 288 | 228 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 411 | 216 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 410 | 215 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 192 | 98 | 94 | 51.04% | 51.04% | 51.04% | 1.04 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 215 | 110 | 105 | 51.16% | 51.16% | 51.16% | 1.16 pp | 5 | 18 | 0.28 |
| BTC Market Hours | nn | NN | 216 | 110 | 106 | 50.93% | 50.93% | 50.93% | 0.93 pp | 4 | 17 | 0.24 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| BTC Market Hours | transformer | Transformer | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 18 | -0.39 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| BTC Market Hours Daily | nn | NN | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 18 | -0.72 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| BTC Market Hours | rf | RandomForest | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 18 | -1.39 |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 10 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 17 | -1.65 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 18 | -2.17 |
| BTC Daily | nn | NN | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| BTC Market Hours | lstm | LSTM | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 17 | -2.24 |
| BTC Hourly | transformer | Transformer | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 8 | -2.25 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 215 | 85 | 130 | 39.53% | 39.53% | 39.53% | 10.47 pp | -45 | 18 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 192 | 80 | 112 | 41.67% | 41.67% | 41.67% | 8.33 pp | -32 | 8 | -4.00 |
| BTC Daily | transformer | Transformer | 218 | 89 | 129 | 40.83% | 40.83% | 40.83% | 9.17 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 218 | 84 | 134 | 38.53% | 38.53% | 38.53% | 11.47 pp | -50 | 10 | -5.00 |
| BTC Hourly | lstm | LSTM | 192 | 72 | 120 | 37.50% | 37.50% | 37.50% | 12.50 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 228 | 81 | 147 | 35.53% | 35.53% | 35.53% | 14.47 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 8 | -6.50 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 192 | 98 | 94 | 51.04% | 51.04% | 51.04% | 1.04 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 8 | -2.25 |
| BTC Hourly | nn | NN | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 192 | 80 | 112 | 41.67% | 41.67% | 41.67% | 8.33 pp | -32 | 8 | -4.00 |
| BTC Hourly | lstm | LSTM | 192 | 72 | 120 | 37.50% | 37.50% | 37.50% | 12.50 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 8 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 10 | -1.40 |
| BTC Daily | nn | NN | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 218 | 89 | 129 | 40.83% | 40.83% | 40.83% | 9.17 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 218 | 84 | 134 | 38.53% | 38.53% | 38.53% | 11.47 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 228 | 81 | 147 | 35.53% | 35.53% | 35.53% | 14.47 pp | -66 | 11 | -6.00 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 216 | 110 | 106 | 50.93% | 50.93% | 50.93% | 0.93 pp | 4 | 17 | 0.24 |
| BTC Market Hours | transformer | Transformer | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 17 | -0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| BTC Market Hours | rf | RandomForest | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 17 | -1.65 |
| BTC Market Hours | lstm | LSTM | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 17 | -2.24 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 215 | 110 | 105 | 51.16% | 51.16% | 51.16% | 1.16 pp | 5 | 18 | 0.28 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 18 | -0.39 |
| BTC Market Hours Daily | nn | NN | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 18 | -0.72 |
| BTC Market Hours Daily | rf | RandomForest | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 18 | -1.39 |
| BTC Market Hours Daily | xgb | XGBoost | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 18 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 215 | 85 | 130 | 39.53% | 39.53% | 39.53% | 10.47 pp | -45 | 18 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
