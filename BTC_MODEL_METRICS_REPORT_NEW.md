# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T04:03:07.638148+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 286 | 226 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 321 | 261 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 470 | 249 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 470 | 249 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 249 | 128 | 121 | 51.41% | 51.67% | 51.41% | 1.41 pp | 7 | 20 | 0.35 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 226 | 114 | 112 | 50.44% | 50.44% | 50.44% | 0.44 pp | 2 | 10 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 249 | 122 | 127 | 49.00% | 49.17% | 49.00% | 1.00 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 249 | 121 | 128 | 48.59% | 47.92% | 48.59% | 1.41 pp | -7 | 21 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| BTC Market Hours Daily | nn | NN | 249 | 119 | 130 | 47.79% | 47.50% | 47.79% | 2.21 pp | -11 | 21 | -0.52 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 249 | 117 | 132 | 46.99% | 47.08% | 46.99% | 3.01 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 249 | 117 | 132 | 46.99% | 46.67% | 46.99% | 3.01 pp | -15 | 20 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 20 | -1.15 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 249 | 112 | 137 | 44.98% | 45.00% | 44.98% | 5.02 pp | -25 | 21 | -1.19 |
| BTC Market Hours Daily | rf | RandomForest | 249 | 110 | 139 | 44.18% | 43.75% | 44.18% | 5.82 pp | -29 | 21 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 251 | 117 | 134 | 46.61% | 46.67% | 46.61% | 3.39 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 20 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 249 | 102 | 147 | 40.96% | 42.08% | 40.96% | 9.04 pp | -45 | 21 | -2.14 |
| BTC Daily | nn | NN | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 11 | -2.27 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| BTC Hourly | transformer | Transformer | 226 | 99 | 127 | 43.81% | 43.81% | 43.81% | 6.19 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 226 | 94 | 132 | 41.59% | 41.59% | 41.59% | 8.41 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 251 | 101 | 150 | 40.24% | 39.58% | 40.24% | 9.76 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 251 | 94 | 157 | 37.45% | 37.08% | 37.45% | 12.55 pp | -63 | 11 | -5.73 |
| BTC Hourly | lstm | LSTM | 226 | 84 | 142 | 37.17% | 37.17% | 37.17% | 12.83 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 261 | 91 | 170 | 34.87% | 35.00% | 34.87% | 15.13 pp | -79 | 12 | -6.58 |
| BTC Hourly | xgb | XGBoost | 226 | 79 | 147 | 34.96% | 34.96% | 34.96% | 15.04 pp | -68 | 10 | -6.80 |
| BTC Daily | lstm | LSTM | 251 | 87 | 164 | 34.66% | 35.42% | 34.66% | 15.34 pp | -77 | 11 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 226 | 114 | 112 | 50.44% | 50.44% | 50.44% | 0.44 pp | 2 | 10 | 0.20 |
| BTC Hourly | transformer | Transformer | 226 | 99 | 127 | 43.81% | 43.81% | 43.81% | 6.19 pp | -28 | 10 | -2.80 |
| BTC Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 226 | 94 | 132 | 41.59% | 41.59% | 41.59% | 8.41 pp | -38 | 10 | -3.80 |
| BTC Hourly | lstm | LSTM | 226 | 84 | 142 | 37.17% | 37.17% | 37.17% | 12.83 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 226 | 79 | 147 | 34.96% | 34.96% | 34.96% | 15.04 pp | -68 | 10 | -6.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 251 | 117 | 134 | 46.61% | 46.67% | 46.61% | 3.39 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 11 | -2.27 |
| BTC Daily | transformer | Transformer | 251 | 101 | 150 | 40.24% | 39.58% | 40.24% | 9.76 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 251 | 94 | 157 | 37.45% | 37.08% | 37.45% | 12.55 pp | -63 | 11 | -5.73 |
| BTC Daily | xgb | XGBoost | 261 | 91 | 170 | 34.87% | 35.00% | 34.87% | 15.13 pp | -79 | 12 | -6.58 |
| BTC Daily | lstm | LSTM | 251 | 87 | 164 | 34.66% | 35.42% | 34.66% | 15.34 pp | -77 | 11 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 249 | 128 | 121 | 51.41% | 51.67% | 51.41% | 1.41 pp | 7 | 20 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 249 | 117 | 132 | 46.99% | 47.08% | 46.99% | 3.01 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 249 | 117 | 132 | 46.99% | 46.67% | 46.99% | 3.01 pp | -15 | 20 | -0.75 |
| BTC Market Hours | rf | RandomForest | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 20 | -1.15 |
| BTC Market Hours | lstm | LSTM | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 20 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 249 | 122 | 127 | 49.00% | 49.17% | 49.00% | 1.00 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 249 | 121 | 128 | 48.59% | 47.92% | 48.59% | 1.41 pp | -7 | 21 | -0.33 |
| BTC Market Hours Daily | nn | NN | 249 | 119 | 130 | 47.79% | 47.50% | 47.79% | 2.21 pp | -11 | 21 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 249 | 112 | 137 | 44.98% | 45.00% | 44.98% | 5.02 pp | -25 | 21 | -1.19 |
| BTC Market Hours Daily | rf | RandomForest | 249 | 110 | 139 | 44.18% | 43.75% | 44.18% | 5.82 pp | -29 | 21 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 249 | 102 | 147 | 40.96% | 42.08% | 40.96% | 9.04 pp | -45 | 21 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
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
