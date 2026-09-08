# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T23:04:40.135008+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 282 | 222 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 318 | 258 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 22:00:00+00:00 | 465 | 246 | 219 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 22:00:00+00:00 | 465 | 246 | 219 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 246 | 126 | 120 | 51.22% | 51.67% | 51.22% | 1.22 pp | 6 | 19 | 0.32 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 222 | 110 | 112 | 49.55% | 49.55% | 49.55% | 0.45 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 246 | 117 | 129 | 47.56% | 47.92% | 47.56% | 2.44 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 246 | 116 | 130 | 47.15% | 47.50% | 47.15% | 2.85 pp | -14 | 19 | -0.74 |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 34 | 39 | 46.58% | 46.58% | 46.58% | 3.42 pp | -5 | 6 | -0.83 |
| BTC Market Hours | xgb | XGBoost | 246 | 115 | 131 | 46.75% | 46.25% | 46.75% | 3.25 pp | -16 | 19 | -0.84 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| BTC Market Hours | transformer | Transformer | 246 | 114 | 132 | 46.34% | 46.67% | 46.34% | 3.66 pp | -18 | 19 | -0.95 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Market Hours | rf | RandomForest | 246 | 111 | 135 | 45.12% | 45.00% | 45.12% | 4.88 pp | -24 | 19 | -1.26 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 246 | 109 | 137 | 44.31% | 43.75% | 44.31% | 5.69 pp | -28 | 20 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 248 | 116 | 132 | 46.77% | 47.08% | 46.77% | 3.23 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 246 | 107 | 139 | 43.50% | 42.50% | 43.50% | 6.50 pp | -32 | 20 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 11 | -2.00 |
| BTC Market Hours | lstm | LSTM | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 19 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 246 | 100 | 146 | 40.65% | 41.25% | 40.65% | 9.35 pp | -46 | 20 | -2.30 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 6 | -2.50 |
| BTC Hourly | transformer | Transformer | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 248 | 99 | 149 | 39.92% | 39.17% | 39.92% | 10.08 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 248 | 95 | 153 | 38.31% | 37.08% | 38.31% | 11.69 pp | -58 | 11 | -5.27 |
| BTC Hourly | lstm | LSTM | 222 | 82 | 140 | 36.94% | 36.94% | 36.94% | 13.06 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 258 | 91 | 167 | 35.27% | 34.58% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 10 | -6.80 |
| BTC Daily | lstm | LSTM | 248 | 85 | 163 | 34.27% | 34.58% | 34.27% | 15.73 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 222 | 110 | 112 | 49.55% | 49.55% | 49.55% | 0.45 pp | -2 | 10 | -0.20 |
| BTC Hourly | transformer | Transformer | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 10 | -3.80 |
| BTC Hourly | lstm | LSTM | 222 | 82 | 140 | 36.94% | 36.94% | 36.94% | 13.06 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 10 | -6.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 248 | 116 | 132 | 46.77% | 47.08% | 46.77% | 3.23 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 248 | 99 | 149 | 39.92% | 39.17% | 39.92% | 10.08 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 248 | 95 | 153 | 38.31% | 37.08% | 38.31% | 11.69 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 258 | 91 | 167 | 35.27% | 34.58% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 248 | 85 | 163 | 34.27% | 34.58% | 34.27% | 15.73 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 246 | 126 | 120 | 51.22% | 51.67% | 51.22% | 1.22 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 246 | 116 | 130 | 47.15% | 47.50% | 47.15% | 2.85 pp | -14 | 19 | -0.74 |
| BTC Market Hours | xgb | XGBoost | 246 | 115 | 131 | 46.75% | 46.25% | 46.75% | 3.25 pp | -16 | 19 | -0.84 |
| BTC Market Hours | transformer | Transformer | 246 | 114 | 132 | 46.34% | 46.67% | 46.34% | 3.66 pp | -18 | 19 | -0.95 |
| BTC Market Hours | rf | RandomForest | 246 | 111 | 135 | 45.12% | 45.00% | 45.12% | 4.88 pp | -24 | 19 | -1.26 |
| BTC Market Hours | lstm | LSTM | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 19 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 246 | 117 | 129 | 47.56% | 47.92% | 47.56% | 2.44 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 246 | 109 | 137 | 44.31% | 43.75% | 44.31% | 5.69 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 246 | 107 | 139 | 43.50% | 42.50% | 43.50% | 6.50 pp | -32 | 20 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 246 | 100 | 146 | 40.65% | 41.25% | 40.65% | 9.35 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 34 | 39 | 46.58% | 46.58% | 46.58% | 3.42 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
