# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T03:43:28.316701+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 253 | 193 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 289 | 229 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 412 | 217 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 412 | 217 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 188 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 217 | 113 | 104 | 52.07% | 52.07% | 52.07% | 2.07 pp | 9 | 17 | 0.53 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | nn | NN | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 217 | 105 | 112 | 48.39% | 48.39% | 48.39% | 1.61 pp | -7 | 18 | -0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 217 | 103 | 114 | 47.47% | 47.47% | 47.47% | 2.53 pp | -11 | 18 | -0.61 |
| BTC Market Hours | rf | RandomForest | 217 | 102 | 115 | 47.00% | 47.00% | 47.00% | 3.00 pp | -13 | 17 | -0.76 |
| Consolidated Hourly | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| BTC Market Hours | transformer | Transformer | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 17 | -1.12 |
| Consolidated Hourly | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 18 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 217 | 93 | 124 | 42.86% | 42.86% | 42.86% | 7.14 pp | -31 | 17 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 217 | 91 | 126 | 41.94% | 41.94% | 41.94% | 8.06 pp | -35 | 18 | -1.94 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 8 | -2.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 217 | 76 | 141 | 35.02% | 35.02% | 35.02% | 14.98 pp | -65 | 18 | -3.61 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 8 | -3.62 |
| BTC Market Hours | lstm | LSTM | 217 | 77 | 140 | 35.48% | 35.48% | 35.48% | 14.52 pp | -63 | 17 | -3.71 |
| BTC Hourly | rf | RandomForest | 193 | 81 | 112 | 41.97% | 41.97% | 41.97% | 8.03 pp | -31 | 8 | -3.88 |
| BTC Daily | transformer | Transformer | 219 | 89 | 130 | 40.64% | 40.64% | 40.64% | 9.36 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 219 | 84 | 135 | 38.36% | 38.36% | 38.36% | 11.64 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 229 | 81 | 148 | 35.37% | 35.37% | 35.37% | 14.63 pp | -67 | 11 | -6.09 |
| BTC Hourly | lstm | LSTM | 193 | 72 | 121 | 37.31% | 37.31% | 37.31% | 12.69 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 8 | -6.62 |
| BTC Daily | lstm | LSTM | 219 | 73 | 146 | 33.33% | 33.33% | 33.33% | 16.67 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 8 | -2.38 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 8 | -3.62 |
| BTC Hourly | rf | RandomForest | 193 | 81 | 112 | 41.97% | 41.97% | 41.97% | 8.03 pp | -31 | 8 | -3.88 |
| BTC Hourly | lstm | LSTM | 193 | 72 | 121 | 37.31% | 37.31% | 37.31% | 12.69 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 8 | -6.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 10 | -1.50 |
| BTC Daily | nn | NN | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| BTC Daily | transformer | Transformer | 219 | 89 | 130 | 40.64% | 40.64% | 40.64% | 9.36 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 219 | 84 | 135 | 38.36% | 38.36% | 38.36% | 11.64 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 229 | 81 | 148 | 35.37% | 35.37% | 35.37% | 14.63 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 219 | 73 | 146 | 33.33% | 33.33% | 33.33% | 16.67 pp | -73 | 10 | -7.30 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 217 | 113 | 104 | 52.07% | 52.07% | 52.07% | 2.07 pp | 9 | 17 | 0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 17 | -0.53 |
| BTC Market Hours | rf | RandomForest | 217 | 102 | 115 | 47.00% | 47.00% | 47.00% | 3.00 pp | -13 | 17 | -0.76 |
| BTC Market Hours | transformer | Transformer | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 17 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 217 | 93 | 124 | 42.86% | 42.86% | 42.86% | 7.14 pp | -31 | 17 | -1.82 |
| BTC Market Hours | lstm | LSTM | 217 | 77 | 140 | 35.48% | 35.48% | 35.48% | 14.52 pp | -63 | 17 | -3.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 217 | 105 | 112 | 48.39% | 48.39% | 48.39% | 1.61 pp | -7 | 18 | -0.39 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 217 | 103 | 114 | 47.47% | 47.47% | 47.47% | 2.53 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 18 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 217 | 91 | 126 | 41.94% | 41.94% | 41.94% | 8.06 pp | -35 | 18 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 217 | 76 | 141 | 35.02% | 35.02% | 35.02% | 14.98 pp | -65 | 18 | -3.61 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
