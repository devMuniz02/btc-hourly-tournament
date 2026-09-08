# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T04:06:10.221872+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 270 | 210 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 306 | 246 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 442 | 234 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 441 | 233 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 234 | 121 | 113 | 51.71% | 51.71% | 51.71% | 1.71 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 210 | 105 | 105 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | transformer | Transformer | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | nn | NN | 233 | 112 | 121 | 48.07% | 48.07% | 48.07% | 1.93 pp | -9 | 19 | -0.47 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 18 | -0.56 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 18 | -1.11 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 233 | 104 | 129 | 44.64% | 44.64% | 44.64% | 5.36 pp | -25 | 19 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 11 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 233 | 101 | 132 | 43.35% | 43.35% | 43.35% | 6.65 pp | -31 | 19 | -1.63 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| BTC Market Hours | lstm | LSTM | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 18 | -2.00 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| BTC Daily | nn | NN | 236 | 106 | 130 | 44.92% | 44.92% | 44.92% | 5.08 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 233 | 94 | 139 | 40.34% | 40.34% | 40.34% | 9.66 pp | -45 | 19 | -2.37 |
| BTC Hourly | transformer | Transformer | 210 | 94 | 116 | 44.76% | 44.76% | 44.76% | 5.24 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 11 | -4.00 |
| BTC Daily | rf | RandomForest | 236 | 91 | 145 | 38.56% | 38.56% | 38.56% | 11.44 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 246 | 88 | 158 | 35.77% | 35.83% | 35.77% | 14.23 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 210 | 78 | 132 | 37.14% | 37.14% | 37.14% | 12.86 pp | -54 | 9 | -6.00 |
| BTC Daily | lstm | LSTM | 236 | 79 | 157 | 33.47% | 33.47% | 33.47% | 16.53 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 210 | 72 | 138 | 34.29% | 34.29% | 34.29% | 15.71 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 210 | 105 | 105 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 210 | 94 | 116 | 44.76% | 44.76% | 44.76% | 5.24 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 210 | 78 | 132 | 37.14% | 37.14% | 37.14% | 12.86 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 210 | 72 | 138 | 34.29% | 34.29% | 34.29% | 15.71 pp | -66 | 9 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 236 | 106 | 130 | 44.92% | 44.92% | 44.92% | 5.08 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 11 | -4.00 |
| BTC Daily | rf | RandomForest | 236 | 91 | 145 | 38.56% | 38.56% | 38.56% | 11.44 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 246 | 88 | 158 | 35.77% | 35.83% | 35.77% | 14.23 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 236 | 79 | 157 | 33.47% | 33.47% | 33.47% | 16.53 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 234 | 121 | 113 | 51.71% | 51.71% | 51.71% | 1.71 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 18 | -1.11 |
| BTC Market Hours | lstm | LSTM | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 18 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | transformer | Transformer | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | nn | NN | 233 | 112 | 121 | 48.07% | 48.07% | 48.07% | 1.93 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | rf | RandomForest | 233 | 104 | 129 | 44.64% | 44.64% | 44.64% | 5.36 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 233 | 101 | 132 | 43.35% | 43.35% | 43.35% | 6.65 pp | -31 | 19 | -1.63 |
| BTC Market Hours Daily | lstm | LSTM | 233 | 94 | 139 | 40.34% | 40.34% | 40.34% | 9.66 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
