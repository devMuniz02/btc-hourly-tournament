# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T12:23:00.591302+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 243 | 183 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 279 | 219 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 389 | 207 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 388 | 206 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 207 | 107 | 100 | 51.69% | 51.69% | 51.69% | 1.69 pp | 7 | 16 | 0.44 |
| BTC Market Hours Daily | transformer | Transformer | 206 | 106 | 100 | 51.46% | 51.46% | 51.46% | 1.46 pp | 6 | 17 | 0.35 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 183 | 92 | 91 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 17 | -0.24 |
| BTC Market Hours | transformer | Transformer | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 16 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | nn | NN | 206 | 98 | 108 | 47.57% | 47.57% | 47.57% | 2.43 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 207 | 97 | 110 | 46.86% | 46.86% | 46.86% | 3.14 pp | -13 | 16 | -0.81 |
| BTC Daily | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 207 | 95 | 112 | 45.89% | 45.89% | 45.89% | 4.11 pp | -17 | 16 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 17 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 16 | -1.56 |
| BTC Hourly | transformer | Transformer | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 8 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 207 | 87 | 120 | 42.03% | 42.03% | 42.03% | 7.97 pp | -33 | 16 | -2.06 |
| BTC Daily | nn | NN | 209 | 94 | 115 | 44.98% | 44.98% | 44.98% | 5.02 pp | -21 | 10 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 206 | 84 | 122 | 40.78% | 40.78% | 40.78% | 9.22 pp | -38 | 17 | -2.24 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 206 | 83 | 123 | 40.29% | 40.29% | 40.29% | 9.71 pp | -40 | 17 | -2.35 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| BTC Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 209 | 87 | 122 | 41.63% | 41.63% | 41.63% | 8.37 pp | -35 | 10 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 183 | 77 | 106 | 42.08% | 42.08% | 42.08% | 7.92 pp | -29 | 8 | -3.62 |
| BTC Daily | rf | RandomForest | 209 | 79 | 130 | 37.80% | 37.80% | 37.80% | 12.20 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 219 | 79 | 140 | 36.07% | 36.07% | 36.07% | 13.93 pp | -61 | 11 | -5.55 |
| BTC Hourly | xgb | XGBoost | 183 | 68 | 115 | 37.16% | 37.16% | 37.16% | 12.84 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 183 | 67 | 116 | 36.61% | 36.61% | 36.61% | 13.39 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 209 | 70 | 139 | 33.49% | 33.49% | 33.49% | 16.51 pp | -69 | 10 | -6.90 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 183 | 92 | 91 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 8 | -1.62 |
| BTC Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 183 | 77 | 106 | 42.08% | 42.08% | 42.08% | 7.92 pp | -29 | 8 | -3.62 |
| BTC Hourly | xgb | XGBoost | 183 | 68 | 115 | 37.16% | 37.16% | 37.16% | 12.84 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 183 | 67 | 116 | 36.61% | 36.61% | 36.61% | 13.39 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 10 | -0.90 |
| BTC Daily | nn | NN | 209 | 94 | 115 | 44.98% | 44.98% | 44.98% | 5.02 pp | -21 | 10 | -2.10 |
| BTC Daily | transformer | Transformer | 209 | 87 | 122 | 41.63% | 41.63% | 41.63% | 8.37 pp | -35 | 10 | -3.50 |
| BTC Daily | rf | RandomForest | 209 | 79 | 130 | 37.80% | 37.80% | 37.80% | 12.20 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 219 | 79 | 140 | 36.07% | 36.07% | 36.07% | 13.93 pp | -61 | 11 | -5.55 |
| BTC Daily | lstm | LSTM | 209 | 70 | 139 | 33.49% | 33.49% | 33.49% | 16.51 pp | -69 | 10 | -6.90 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 207 | 107 | 100 | 51.69% | 51.69% | 51.69% | 1.69 pp | 7 | 16 | 0.44 |
| BTC Market Hours | transformer | Transformer | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 16 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 207 | 97 | 110 | 46.86% | 46.86% | 46.86% | 3.14 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 207 | 95 | 112 | 45.89% | 45.89% | 45.89% | 4.11 pp | -17 | 16 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 16 | -1.56 |
| BTC Market Hours | lstm | LSTM | 207 | 87 | 120 | 42.03% | 42.03% | 42.03% | 7.97 pp | -33 | 16 | -2.06 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 206 | 106 | 100 | 51.46% | 51.46% | 51.46% | 1.46 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 17 | -0.24 |
| BTC Market Hours Daily | nn | NN | 206 | 98 | 108 | 47.57% | 47.57% | 47.57% | 2.43 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 206 | 84 | 122 | 40.78% | 40.78% | 40.78% | 9.22 pp | -38 | 17 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 206 | 83 | 123 | 40.29% | 40.29% | 40.29% | 9.71 pp | -40 | 17 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
