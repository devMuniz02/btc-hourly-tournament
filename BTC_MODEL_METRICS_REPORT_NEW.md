# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T22:29:04.080262+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 201 | 141 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 236 | 176 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 317 | 164 | 153 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 317 | 164 | 153 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 141 | 73 | 68 | 51.77% | 51.77% | 51.77% | 1.77 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 13 | 0.31 |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 6 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 13 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 13 | -2.31 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 6 | -3.17 |
| BTC Hourly | rf | RandomForest | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 6 | -3.83 |
| BTC Daily | rf | RandomForest | 166 | 66 | 100 | 39.76% | 39.76% | 39.76% | 10.24 pp | -34 | 8 | -4.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 166 | 58 | 108 | 34.94% | 34.94% | 34.94% | 15.06 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 141 | 51 | 90 | 36.17% | 36.17% | 36.17% | 13.83 pp | -39 | 6 | -6.50 |
| BTC Hourly | lstm | LSTM | 141 | 50 | 91 | 35.46% | 35.46% | 35.46% | 14.54 pp | -41 | 6 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 141 | 73 | 68 | 51.77% | 51.77% | 51.77% | 1.77 pp | 5 | 6 | 0.83 |
| BTC Hourly | transformer | Transformer | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 6 | -3.17 |
| BTC Hourly | rf | RandomForest | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 6 | -3.83 |
| BTC Hourly | xgb | XGBoost | 141 | 51 | 90 | 36.17% | 36.17% | 36.17% | 13.83 pp | -39 | 6 | -6.50 |
| BTC Hourly | lstm | LSTM | 141 | 50 | 91 | 35.46% | 35.46% | 35.46% | 14.54 pp | -41 | 6 | -6.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 166 | 66 | 100 | 39.76% | 39.76% | 39.76% | 10.24 pp | -34 | 8 | -4.25 |
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
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
