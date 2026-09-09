# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T05:58:21.876877+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 287 | 227 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 323 | 263 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 472 | 251 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 472 | 251 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 251 | 130 | 121 | 51.79% | 52.08% | 51.79% | 1.79 pp | 9 | 20 | 0.45 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 227 | 114 | 113 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 251 | 124 | 127 | 49.40% | 49.17% | 49.40% | 0.60 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 251 | 123 | 128 | 49.00% | 48.33% | 49.00% | 1.00 pp | -5 | 21 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 251 | 120 | 131 | 47.81% | 47.08% | 47.81% | 2.19 pp | -11 | 21 | -0.52 |
| BTC Market Hours | transformer | Transformer | 251 | 119 | 132 | 47.41% | 47.92% | 47.41% | 2.59 pp | -13 | 20 | -0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 251 | 118 | 133 | 47.01% | 47.08% | 47.01% | 2.99 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 251 | 117 | 134 | 46.61% | 46.25% | 46.61% | 3.39 pp | -17 | 20 | -0.85 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 20 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 251 | 112 | 139 | 44.62% | 45.00% | 44.62% | 5.38 pp | -27 | 21 | -1.29 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 251 | 110 | 141 | 43.82% | 43.33% | 43.82% | 6.18 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours | lstm | LSTM | 251 | 105 | 146 | 41.83% | 42.50% | 41.83% | 8.17 pp | -41 | 20 | -2.05 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| BTC Daily | nn | NN | 253 | 115 | 138 | 45.45% | 44.17% | 45.45% | 4.55 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 21 | -2.14 |
| BTC Hourly | transformer | Transformer | 227 | 100 | 127 | 44.05% | 44.05% | 44.05% | 5.95 pp | -27 | 10 | -2.70 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 227 | 94 | 133 | 41.41% | 41.41% | 41.41% | 8.59 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 253 | 102 | 151 | 40.32% | 39.17% | 40.32% | 9.68 pp | -49 | 11 | -4.45 |
| BTC Hourly | lstm | LSTM | 227 | 85 | 142 | 37.44% | 37.44% | 37.44% | 12.56 pp | -57 | 10 | -5.70 |
| BTC Daily | rf | RandomForest | 253 | 95 | 158 | 37.55% | 36.67% | 37.55% | 12.45 pp | -63 | 11 | -5.73 |
| BTC Daily | xgb | XGBoost | 263 | 93 | 170 | 35.36% | 35.00% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 227 | 79 | 148 | 34.80% | 34.80% | 34.80% | 15.20 pp | -69 | 10 | -6.90 |
| BTC Daily | lstm | LSTM | 253 | 88 | 165 | 34.78% | 35.42% | 34.78% | 15.22 pp | -77 | 11 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 227 | 114 | 113 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Hourly | transformer | Transformer | 227 | 100 | 127 | 44.05% | 44.05% | 44.05% | 5.95 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 227 | 94 | 133 | 41.41% | 41.41% | 41.41% | 8.59 pp | -39 | 10 | -3.90 |
| BTC Hourly | lstm | LSTM | 227 | 85 | 142 | 37.44% | 37.44% | 37.44% | 12.56 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 227 | 79 | 148 | 34.80% | 34.80% | 34.80% | 15.20 pp | -69 | 10 | -6.90 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 253 | 115 | 138 | 45.45% | 44.17% | 45.45% | 4.55 pp | -23 | 11 | -2.09 |
| BTC Daily | transformer | Transformer | 253 | 102 | 151 | 40.32% | 39.17% | 40.32% | 9.68 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 253 | 95 | 158 | 37.55% | 36.67% | 37.55% | 12.45 pp | -63 | 11 | -5.73 |
| BTC Daily | xgb | XGBoost | 263 | 93 | 170 | 35.36% | 35.00% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |
| BTC Daily | lstm | LSTM | 253 | 88 | 165 | 34.78% | 35.42% | 34.78% | 15.22 pp | -77 | 11 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 251 | 130 | 121 | 51.79% | 52.08% | 51.79% | 1.79 pp | 9 | 20 | 0.45 |
| BTC Market Hours | transformer | Transformer | 251 | 119 | 132 | 47.41% | 47.92% | 47.41% | 2.59 pp | -13 | 20 | -0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 251 | 118 | 133 | 47.01% | 47.08% | 47.01% | 2.99 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 251 | 117 | 134 | 46.61% | 46.25% | 46.61% | 3.39 pp | -17 | 20 | -0.85 |
| BTC Market Hours | rf | RandomForest | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 20 | -1.25 |
| BTC Market Hours | lstm | LSTM | 251 | 105 | 146 | 41.83% | 42.50% | 41.83% | 8.17 pp | -41 | 20 | -2.05 |

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
