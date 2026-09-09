# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T07:14:39.005864+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 288 | 228 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 324 | 264 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 473 | 252 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 472 | 251 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 252 | 131 | 121 | 51.98% | 52.08% | 51.98% | 1.98 pp | 10 | 20 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 228 | 114 | 114 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 251 | 124 | 127 | 49.40% | 49.17% | 49.40% | 0.60 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 251 | 123 | 128 | 49.00% | 48.33% | 49.00% | 1.00 pp | -5 | 21 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 251 | 120 | 131 | 47.81% | 47.08% | 47.81% | 2.19 pp | -11 | 21 | -0.52 |
| BTC Market Hours | transformer | Transformer | 252 | 120 | 132 | 47.62% | 47.92% | 47.62% | 2.38 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 252 | 117 | 135 | 46.43% | 45.83% | 46.43% | 3.57 pp | -18 | 20 | -0.90 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 251 | 112 | 139 | 44.62% | 45.00% | 44.62% | 5.38 pp | -27 | 21 | -1.29 |
| BTC Market Hours | rf | RandomForest | 252 | 113 | 139 | 44.84% | 44.58% | 44.84% | 5.16 pp | -26 | 20 | -1.30 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 254 | 119 | 135 | 46.85% | 46.67% | 46.85% | 3.15 pp | -16 | 11 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 251 | 110 | 141 | 43.82% | 43.33% | 43.82% | 6.18 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 254 | 116 | 138 | 45.67% | 44.58% | 45.67% | 4.33 pp | -22 | 11 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| BTC Market Hours | lstm | LSTM | 252 | 105 | 147 | 41.67% | 42.08% | 41.67% | 8.33 pp | -42 | 20 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 21 | -2.14 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| BTC Hourly | transformer | Transformer | 228 | 100 | 128 | 43.86% | 43.86% | 43.86% | 6.14 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 228 | 95 | 133 | 41.67% | 41.67% | 41.67% | 8.33 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 10 | -4.00 |
| BTC Daily | transformer | Transformer | 254 | 103 | 151 | 40.55% | 39.58% | 40.55% | 9.45 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 254 | 96 | 158 | 37.80% | 37.08% | 37.80% | 12.20 pp | -62 | 11 | -5.64 |
| BTC Hourly | lstm | LSTM | 228 | 85 | 143 | 37.28% | 37.28% | 37.28% | 12.72 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 264 | 94 | 170 | 35.61% | 35.42% | 35.61% | 14.39 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 228 | 79 | 149 | 34.65% | 34.65% | 34.65% | 15.35 pp | -70 | 10 | -7.00 |
| BTC Daily | lstm | LSTM | 254 | 88 | 166 | 34.65% | 35.42% | 34.65% | 15.35 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 228 | 114 | 114 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Hourly | transformer | Transformer | 228 | 100 | 128 | 43.86% | 43.86% | 43.86% | 6.14 pp | -28 | 10 | -2.80 |
| BTC Hourly | nn | NN | 228 | 95 | 133 | 41.67% | 41.67% | 41.67% | 8.33 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 10 | -4.00 |
| BTC Hourly | lstm | LSTM | 228 | 85 | 143 | 37.28% | 37.28% | 37.28% | 12.72 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 228 | 79 | 149 | 34.65% | 34.65% | 34.65% | 15.35 pp | -70 | 10 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 254 | 119 | 135 | 46.85% | 46.67% | 46.85% | 3.15 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 254 | 116 | 138 | 45.67% | 44.58% | 45.67% | 4.33 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 254 | 103 | 151 | 40.55% | 39.58% | 40.55% | 9.45 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 254 | 96 | 158 | 37.80% | 37.08% | 37.80% | 12.20 pp | -62 | 11 | -5.64 |
| BTC Daily | xgb | XGBoost | 264 | 94 | 170 | 35.61% | 35.42% | 35.61% | 14.39 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 254 | 88 | 166 | 34.65% | 35.42% | 34.65% | 15.35 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 252 | 131 | 121 | 51.98% | 52.08% | 51.98% | 1.98 pp | 10 | 20 | 0.50 |
| BTC Market Hours | transformer | Transformer | 252 | 120 | 132 | 47.62% | 47.92% | 47.62% | 2.38 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 252 | 117 | 135 | 46.43% | 45.83% | 46.43% | 3.57 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 252 | 113 | 139 | 44.84% | 44.58% | 44.84% | 5.16 pp | -26 | 20 | -1.30 |
| BTC Market Hours | lstm | LSTM | 252 | 105 | 147 | 41.67% | 42.08% | 41.67% | 8.33 pp | -42 | 20 | -2.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 251 | 124 | 127 | 49.40% | 49.17% | 49.40% | 0.60 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 251 | 123 | 128 | 49.00% | 48.33% | 49.00% | 1.00 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | nn | NN | 251 | 120 | 131 | 47.81% | 47.08% | 47.81% | 2.19 pp | -11 | 21 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 251 | 112 | 139 | 44.62% | 45.00% | 44.62% | 5.38 pp | -27 | 21 | -1.29 |
| BTC Market Hours Daily | rf | RandomForest | 251 | 110 | 141 | 43.82% | 43.33% | 43.82% | 6.18 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | lstm | LSTM | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 21 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
