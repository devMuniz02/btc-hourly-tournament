# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T04:20:56.127355+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 217 | 113 | 104 | 52.07% | 52.07% | 52.07% | 2.07 pp | 9 | 17 | 0.53 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 217 | 105 | 112 | 48.39% | 48.39% | 48.39% | 1.61 pp | -7 | 18 | -0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 217 | 103 | 114 | 47.47% | 47.47% | 47.47% | 2.53 pp | -11 | 18 | -0.61 |
| BTC Market Hours | rf | RandomForest | 217 | 102 | 115 | 47.00% | 47.00% | 47.00% | 3.00 pp | -13 | 17 | -0.76 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| BTC Market Hours | transformer | Transformer | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 17 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 18 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 217 | 93 | 124 | 42.86% | 42.86% | 42.86% | 7.14 pp | -31 | 17 | -1.82 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 217 | 91 | 126 | 41.94% | 41.94% | 41.94% | 8.06 pp | -35 | 18 | -1.94 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 9 | -2.11 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 9 | -3.22 |
| BTC Hourly | rf | RandomForest | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 9 | -3.22 |
| BTC Market Hours Daily | lstm | LSTM | 217 | 76 | 141 | 35.02% | 35.02% | 35.02% | 14.98 pp | -65 | 18 | -3.61 |
| BTC Market Hours | lstm | LSTM | 217 | 77 | 140 | 35.48% | 35.48% | 35.48% | 14.52 pp | -63 | 17 | -3.71 |
| BTC Daily | transformer | Transformer | 219 | 89 | 130 | 40.64% | 40.64% | 40.64% | 9.36 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 219 | 84 | 135 | 38.36% | 38.36% | 38.36% | 11.64 pp | -51 | 10 | -5.10 |
| BTC Hourly | lstm | LSTM | 193 | 73 | 120 | 37.82% | 37.82% | 37.82% | 12.18 pp | -47 | 9 | -5.22 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 229 | 81 | 148 | 35.37% | 35.37% | 35.37% | 14.63 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 219 | 73 | 146 | 33.33% | 33.33% | 33.33% | 16.67 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 9 | 0.33 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 9 | -2.11 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 9 | -3.22 |
| BTC Hourly | rf | RandomForest | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 9 | -3.22 |
| BTC Hourly | lstm | LSTM | 193 | 73 | 120 | 37.82% | 37.82% | 37.82% | 12.18 pp | -47 | 9 | -5.22 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 9 | -5.89 |

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
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
