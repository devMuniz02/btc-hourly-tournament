# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T05:39:34.538109+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 322 | 262 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 471 | 250 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 471 | 250 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 250 | 129 | 121 | 51.60% | 51.67% | 51.60% | 1.60 pp | 8 | 20 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 227 | 114 | 113 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 250 | 123 | 127 | 49.20% | 49.17% | 49.20% | 0.80 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 250 | 122 | 128 | 48.80% | 47.92% | 48.80% | 1.20 pp | -6 | 21 | -0.29 |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 250 | 119 | 131 | 47.60% | 47.08% | 47.60% | 2.40 pp | -12 | 21 | -0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 250 | 118 | 132 | 47.20% | 47.50% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 250 | 117 | 133 | 46.80% | 46.25% | 46.80% | 3.20 pp | -16 | 20 | -0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 250 | 113 | 137 | 45.20% | 45.00% | 45.20% | 4.80 pp | -24 | 20 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 250 | 112 | 138 | 44.80% | 45.00% | 44.80% | 5.20 pp | -26 | 21 | -1.24 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 250 | 110 | 140 | 44.00% | 43.75% | 44.00% | 6.00 pp | -30 | 21 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 250 | 105 | 145 | 42.00% | 42.50% | 42.00% | 8.00 pp | -40 | 20 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 250 | 103 | 147 | 41.20% | 42.08% | 41.20% | 8.80 pp | -44 | 21 | -2.10 |
| BTC Daily | nn | NN | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| BTC Hourly | transformer | Transformer | 227 | 100 | 127 | 44.05% | 44.05% | 44.05% | 5.95 pp | -27 | 10 | -2.70 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 227 | 94 | 133 | 41.41% | 41.41% | 41.41% | 8.59 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 252 | 101 | 151 | 40.08% | 39.17% | 40.08% | 9.92 pp | -50 | 11 | -4.55 |
| BTC Hourly | lstm | LSTM | 227 | 85 | 142 | 37.44% | 37.44% | 37.44% | 12.56 pp | -57 | 10 | -5.70 |
| BTC Daily | rf | RandomForest | 252 | 94 | 158 | 37.30% | 36.67% | 37.30% | 12.70 pp | -64 | 11 | -5.82 |
| BTC Daily | xgb | XGBoost | 262 | 92 | 170 | 35.11% | 35.00% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 227 | 79 | 148 | 34.80% | 34.80% | 34.80% | 15.20 pp | -69 | 10 | -6.90 |
| BTC Daily | lstm | LSTM | 252 | 88 | 164 | 34.92% | 35.42% | 34.92% | 15.08 pp | -76 | 11 | -6.91 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 252 | 101 | 151 | 40.08% | 39.17% | 40.08% | 9.92 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 252 | 94 | 158 | 37.30% | 36.67% | 37.30% | 12.70 pp | -64 | 11 | -5.82 |
| BTC Daily | xgb | XGBoost | 262 | 92 | 170 | 35.11% | 35.00% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |
| BTC Daily | lstm | LSTM | 252 | 88 | 164 | 34.92% | 35.42% | 34.92% | 15.08 pp | -76 | 11 | -6.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 250 | 129 | 121 | 51.60% | 51.67% | 51.60% | 1.60 pp | 8 | 20 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 250 | 118 | 132 | 47.20% | 47.50% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 250 | 117 | 133 | 46.80% | 46.25% | 46.80% | 3.20 pp | -16 | 20 | -0.80 |
| BTC Market Hours | rf | RandomForest | 250 | 113 | 137 | 45.20% | 45.00% | 45.20% | 4.80 pp | -24 | 20 | -1.20 |
| BTC Market Hours | lstm | LSTM | 250 | 105 | 145 | 42.00% | 42.50% | 42.00% | 8.00 pp | -40 | 20 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 250 | 123 | 127 | 49.20% | 49.17% | 49.20% | 0.80 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 250 | 122 | 128 | 48.80% | 47.92% | 48.80% | 1.20 pp | -6 | 21 | -0.29 |
| BTC Market Hours Daily | nn | NN | 250 | 119 | 131 | 47.60% | 47.08% | 47.60% | 2.40 pp | -12 | 21 | -0.57 |
| BTC Market Hours Daily | xgb | XGBoost | 250 | 112 | 138 | 44.80% | 45.00% | 44.80% | 5.20 pp | -26 | 21 | -1.24 |
| BTC Market Hours Daily | rf | RandomForest | 250 | 110 | 140 | 44.00% | 43.75% | 44.00% | 6.00 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | lstm | LSTM | 250 | 103 | 147 | 41.20% | 42.08% | 41.20% | 8.80 pp | -44 | 21 | -2.10 |

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
