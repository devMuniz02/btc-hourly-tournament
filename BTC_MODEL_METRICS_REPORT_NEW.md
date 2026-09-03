# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T22:04:49.331804+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 200 | 140 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 236 | 176 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 317 | 164 | 153 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 317 | 164 | 153 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 140 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 6 | 0.67 |
| BTC Market Hours | nn | NN | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 13 | 0.31 |
| Consolidated Hourly | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 6 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 13 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| BTC Market Hours | lstm | LSTM | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 13 | -2.31 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |
| BTC Hourly | nn | NN | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 6 | -3.00 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 166 | 67 | 99 | 40.36% | 40.36% | 40.36% | 9.64 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 166 | 58 | 108 | 34.94% | 34.94% | 34.94% | 15.06 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 140 | 50 | 90 | 35.71% | 35.71% | 35.71% | 14.29 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 6 | 0.67 |
| BTC Hourly | transformer | Transformer | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 6 | -3.00 |
| BTC Hourly | rf | RandomForest | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 140 | 50 | 90 | 35.71% | 35.71% | 35.71% | 14.29 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 166 | 67 | 99 | 40.36% | 40.36% | 40.36% | 9.64 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 166 | 58 | 108 | 34.94% | 34.94% | 34.94% | 15.06 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 13 | 0.31 |
| BTC Market Hours | rf | RandomForest | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 13 | -0.92 |
| BTC Market Hours | transformer | Transformer | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
