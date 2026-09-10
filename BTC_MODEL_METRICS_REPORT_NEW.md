# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T00:15:44.386120+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 299 | 239 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 335 | 275 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 23:00:00+00:00 | 496 | 263 | 233 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 23:00:00+00:00 | 495 | 262 | 233 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 263 | 139 | 124 | 52.85% | 52.92% | 52.85% | 2.85 pp | 15 | 21 | 0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 262 | 130 | 132 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 262 | 129 | 133 | 49.24% | 49.17% | 49.24% | 0.76 pp | -4 | 22 | -0.18 |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 262 | 126 | 136 | 48.09% | 48.33% | 48.09% | 1.91 pp | -10 | 22 | -0.45 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 239 | 117 | 122 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 10 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 263 | 124 | 139 | 47.15% | 48.33% | 47.15% | 2.85 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 263 | 123 | 140 | 46.77% | 47.08% | 46.77% | 3.23 pp | -17 | 21 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 263 | 121 | 142 | 46.01% | 44.58% | 46.01% | 3.99 pp | -21 | 21 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 262 | 115 | 147 | 43.89% | 42.92% | 43.89% | 6.11 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 263 | 116 | 147 | 44.11% | 43.33% | 44.11% | 5.89 pp | -31 | 21 | -1.48 |
| BTC Daily | nn | NN | 265 | 123 | 142 | 46.42% | 45.42% | 46.42% | 3.58 pp | -19 | 12 | -1.58 |
| BTC Market Hours Daily | rf | RandomForest | 262 | 113 | 149 | 43.13% | 42.08% | 43.13% | 6.87 pp | -36 | 22 | -1.64 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 265 | 122 | 143 | 46.04% | 45.00% | 46.04% | 3.96 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 262 | 108 | 154 | 41.22% | 42.08% | 41.22% | 8.78 pp | -46 | 22 | -2.09 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| BTC Market Hours | lstm | LSTM | 263 | 107 | 156 | 40.68% | 42.08% | 40.68% | 9.32 pp | -49 | 21 | -2.33 |
| BTC Hourly | transformer | Transformer | 239 | 107 | 132 | 44.77% | 44.77% | 44.77% | 5.23 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| BTC Hourly | nn | NN | 239 | 101 | 138 | 42.26% | 42.26% | 42.26% | 7.74 pp | -37 | 10 | -3.70 |
| BTC Daily | transformer | Transformer | 265 | 108 | 157 | 40.75% | 40.00% | 40.75% | 9.25 pp | -49 | 12 | -4.08 |
| BTC Hourly | rf | RandomForest | 239 | 98 | 141 | 41.00% | 41.00% | 41.00% | 9.00 pp | -43 | 10 | -4.30 |
| BTC Daily | rf | RandomForest | 265 | 100 | 165 | 37.74% | 37.50% | 37.74% | 12.26 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 275 | 101 | 174 | 36.73% | 37.08% | 36.73% | 13.27 pp | -73 | 13 | -5.62 |
| BTC Hourly | lstm | LSTM | 239 | 88 | 151 | 36.82% | 36.82% | 36.82% | 13.18 pp | -63 | 10 | -6.30 |
| BTC Daily | lstm | LSTM | 265 | 94 | 171 | 35.47% | 36.25% | 35.47% | 14.53 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 239 | 82 | 157 | 34.31% | 34.31% | 34.31% | 15.69 pp | -75 | 10 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 239 | 117 | 122 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 10 | -0.50 |
| BTC Hourly | transformer | Transformer | 239 | 107 | 132 | 44.77% | 44.77% | 44.77% | 5.23 pp | -25 | 10 | -2.50 |
| BTC Hourly | nn | NN | 239 | 101 | 138 | 42.26% | 42.26% | 42.26% | 7.74 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 239 | 98 | 141 | 41.00% | 41.00% | 41.00% | 9.00 pp | -43 | 10 | -4.30 |
| BTC Hourly | lstm | LSTM | 239 | 88 | 151 | 36.82% | 36.82% | 36.82% | 13.18 pp | -63 | 10 | -6.30 |
| BTC Hourly | xgb | XGBoost | 239 | 82 | 157 | 34.31% | 34.31% | 34.31% | 15.69 pp | -75 | 10 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 265 | 123 | 142 | 46.42% | 45.42% | 46.42% | 3.58 pp | -19 | 12 | -1.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 265 | 122 | 143 | 46.04% | 45.00% | 46.04% | 3.96 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 265 | 108 | 157 | 40.75% | 40.00% | 40.75% | 9.25 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 265 | 100 | 165 | 37.74% | 37.50% | 37.74% | 12.26 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 275 | 101 | 174 | 36.73% | 37.08% | 36.73% | 13.27 pp | -73 | 13 | -5.62 |
| BTC Daily | lstm | LSTM | 265 | 94 | 171 | 35.47% | 36.25% | 35.47% | 14.53 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 263 | 139 | 124 | 52.85% | 52.92% | 52.85% | 2.85 pp | 15 | 21 | 0.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 263 | 124 | 139 | 47.15% | 48.33% | 47.15% | 2.85 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 263 | 123 | 140 | 46.77% | 47.08% | 46.77% | 3.23 pp | -17 | 21 | -0.81 |
| BTC Market Hours | xgb | XGBoost | 263 | 121 | 142 | 46.01% | 44.58% | 46.01% | 3.99 pp | -21 | 21 | -1.00 |
| BTC Market Hours | rf | RandomForest | 263 | 116 | 147 | 44.11% | 43.33% | 44.11% | 5.89 pp | -31 | 21 | -1.48 |
| BTC Market Hours | lstm | LSTM | 263 | 107 | 156 | 40.68% | 42.08% | 40.68% | 9.32 pp | -49 | 21 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 262 | 130 | 132 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 262 | 129 | 133 | 49.24% | 49.17% | 49.24% | 0.76 pp | -4 | 22 | -0.18 |
| BTC Market Hours Daily | nn | NN | 262 | 126 | 136 | 48.09% | 48.33% | 48.09% | 1.91 pp | -10 | 22 | -0.45 |
| BTC Market Hours Daily | xgb | XGBoost | 262 | 115 | 147 | 43.89% | 42.92% | 43.89% | 6.11 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 262 | 113 | 149 | 43.13% | 42.08% | 43.13% | 6.87 pp | -36 | 22 | -1.64 |
| BTC Market Hours Daily | lstm | LSTM | 262 | 108 | 154 | 41.22% | 42.08% | 41.22% | 8.78 pp | -46 | 22 | -2.09 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
