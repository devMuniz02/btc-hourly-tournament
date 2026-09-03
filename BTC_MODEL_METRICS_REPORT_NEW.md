# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T22:38:56.917047+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 237 | 177 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 318 | 165 | 153 | 0 |
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
| BTC Market Hours | nn | NN | 165 | 85 | 80 | 51.52% | 51.52% | 51.52% | 1.52 pp | 5 | 13 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 6 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 165 | 77 | 88 | 46.67% | 46.67% | 46.67% | 3.33 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| BTC Market Hours | transformer | Transformer | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 13 | -1.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 167 | 78 | 89 | 46.71% | 46.71% | 46.71% | 3.29 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 165 | 68 | 97 | 41.21% | 41.21% | 41.21% | 8.79 pp | -29 | 13 | -2.23 |
| BTC Daily | nn | NN | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |
| BTC Daily | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 6 | -3.17 |
| BTC Hourly | rf | RandomForest | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 6 | -3.83 |
| BTC Daily | rf | RandomForest | 167 | 67 | 100 | 40.12% | 40.12% | 40.12% | 9.88 pp | -33 | 8 | -4.12 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 167 | 58 | 109 | 34.73% | 34.73% | 34.73% | 15.27 pp | -51 | 8 | -6.38 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 167 | 78 | 89 | 46.71% | 46.71% | 46.71% | 3.29 pp | -11 | 8 | -1.38 |
| BTC Daily | nn | NN | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 8 | -2.38 |
| BTC Daily | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 167 | 67 | 100 | 40.12% | 40.12% | 40.12% | 9.88 pp | -33 | 8 | -4.12 |
| BTC Daily | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 167 | 58 | 109 | 34.73% | 34.73% | 34.73% | 15.27 pp | -51 | 8 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 165 | 85 | 80 | 51.52% | 51.52% | 51.52% | 1.52 pp | 5 | 13 | 0.38 |
| BTC Market Hours | rf | RandomForest | 165 | 77 | 88 | 46.67% | 46.67% | 46.67% | 3.33 pp | -11 | 13 | -0.85 |
| BTC Market Hours | transformer | Transformer | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 13 | -1.92 |
| BTC Market Hours | lstm | LSTM | 165 | 68 | 97 | 41.21% | 41.21% | 41.21% | 8.79 pp | -29 | 13 | -2.23 |

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
