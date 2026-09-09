# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T02:41:24.914881+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 285 | 225 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 320 | 260 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 469 | 248 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 469 | 248 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 248 | 128 | 120 | 51.61% | 52.08% | 51.61% | 1.61 pp | 8 | 20 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 225 | 113 | 112 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 248 | 121 | 127 | 48.79% | 48.33% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 248 | 121 | 127 | 48.79% | 49.17% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 248 | 118 | 130 | 47.58% | 47.92% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | nn | NN | 248 | 118 | 130 | 47.58% | 47.50% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours | xgb | XGBoost | 248 | 117 | 131 | 47.18% | 47.08% | 47.18% | 2.82 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 248 | 116 | 132 | 46.77% | 46.67% | 46.77% | 3.23 pp | -16 | 20 | -0.80 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 20 | -1.10 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 248 | 111 | 137 | 44.76% | 44.58% | 44.76% | 5.24 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 248 | 109 | 139 | 43.95% | 43.33% | 43.95% | 6.05 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 250 | 116 | 134 | 46.40% | 46.67% | 46.40% | 3.60 pp | -18 | 11 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 248 | 105 | 143 | 42.34% | 43.33% | 42.34% | 7.66 pp | -38 | 20 | -1.90 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 248 | 101 | 147 | 40.73% | 41.67% | 40.73% | 9.27 pp | -46 | 20 | -2.30 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| BTC Hourly | transformer | Transformer | 225 | 99 | 126 | 44.00% | 44.00% | 44.00% | 6.00 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 225 | 93 | 132 | 41.33% | 41.33% | 41.33% | 8.67 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 250 | 100 | 150 | 40.00% | 39.58% | 40.00% | 10.00 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 250 | 94 | 156 | 37.60% | 37.08% | 37.60% | 12.40 pp | -62 | 11 | -5.64 |
| BTC Hourly | lstm | LSTM | 225 | 84 | 141 | 37.33% | 37.33% | 37.33% | 12.67 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 260 | 91 | 169 | 35.00% | 35.00% | 35.00% | 15.00 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 225 | 79 | 146 | 35.11% | 35.11% | 35.11% | 14.89 pp | -67 | 10 | -6.70 |
| BTC Daily | lstm | LSTM | 250 | 86 | 164 | 34.40% | 35.00% | 34.40% | 15.60 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 225 | 113 | 112 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Hourly | transformer | Transformer | 225 | 99 | 126 | 44.00% | 44.00% | 44.00% | 6.00 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 225 | 93 | 132 | 41.33% | 41.33% | 41.33% | 8.67 pp | -39 | 10 | -3.90 |
| BTC Hourly | lstm | LSTM | 225 | 84 | 141 | 37.33% | 37.33% | 37.33% | 12.67 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 225 | 79 | 146 | 35.11% | 35.11% | 35.11% | 14.89 pp | -67 | 10 | -6.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 250 | 116 | 134 | 46.40% | 46.67% | 46.40% | 3.60 pp | -18 | 11 | -1.64 |
| BTC Daily | nn | NN | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 250 | 100 | 150 | 40.00% | 39.58% | 40.00% | 10.00 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 250 | 94 | 156 | 37.60% | 37.08% | 37.60% | 12.40 pp | -62 | 11 | -5.64 |
| BTC Daily | xgb | XGBoost | 260 | 91 | 169 | 35.00% | 35.00% | 35.00% | 15.00 pp | -78 | 12 | -6.50 |
| BTC Daily | lstm | LSTM | 250 | 86 | 164 | 34.40% | 35.00% | 34.40% | 15.60 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 248 | 128 | 120 | 51.61% | 52.08% | 51.61% | 1.61 pp | 8 | 20 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 248 | 118 | 130 | 47.58% | 47.92% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours | xgb | XGBoost | 248 | 117 | 131 | 47.18% | 47.08% | 47.18% | 2.82 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 248 | 116 | 132 | 46.77% | 46.67% | 46.77% | 3.23 pp | -16 | 20 | -0.80 |
| BTC Market Hours | rf | RandomForest | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 20 | -1.10 |
| BTC Market Hours | lstm | LSTM | 248 | 105 | 143 | 42.34% | 43.33% | 42.34% | 7.66 pp | -38 | 20 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 248 | 121 | 127 | 48.79% | 48.33% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 248 | 121 | 127 | 48.79% | 49.17% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 248 | 118 | 130 | 47.58% | 47.50% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 248 | 111 | 137 | 44.76% | 44.58% | 44.76% | 5.24 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 248 | 109 | 139 | 43.95% | 43.33% | 43.95% | 6.05 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 248 | 101 | 147 | 40.73% | 41.67% | 40.73% | 9.27 pp | -46 | 20 | -2.30 |

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
