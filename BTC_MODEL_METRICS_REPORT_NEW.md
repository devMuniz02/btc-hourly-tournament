# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T04:32:57.092441+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 322 | 262 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 471 | 250 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 471 | 250 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 218 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 218 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 218 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 219 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 250 | 129 | 121 | 51.60% | 51.67% | 51.60% | 1.60 pp | 8 | 20 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 226 | 114 | 112 | 50.44% | 50.44% | 50.44% | 0.44 pp | 2 | 10 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 250 | 123 | 127 | 49.20% | 49.17% | 49.20% | 0.80 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 250 | 122 | 128 | 48.80% | 47.92% | 48.80% | 1.20 pp | -6 | 21 | -0.29 |
| BTC Market Hours Daily | nn | NN | 250 | 119 | 131 | 47.60% | 47.08% | 47.60% | 2.40 pp | -12 | 21 | -0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 250 | 118 | 132 | 47.20% | 47.50% | 47.20% | 2.80 pp | -14 | 20 | -0.70 |
| Consolidated Hourly | rf | RandomForest | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 14 | -0.71 |
| BTC Market Hours | xgb | XGBoost | 250 | 117 | 133 | 46.80% | 46.25% | 46.80% | 3.20 pp | -16 | 20 | -0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Market Hours | rf | RandomForest | 250 | 113 | 137 | 45.20% | 45.00% | 45.20% | 4.80 pp | -24 | 20 | -1.20 |
| BTC Market Hours Daily | xgb | XGBoost | 250 | 112 | 138 | 44.80% | 45.00% | 44.80% | 5.20 pp | -26 | 21 | -1.24 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | xgb | XGBoost | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 250 | 110 | 140 | 44.00% | 43.75% | 44.00% | 6.00 pp | -30 | 21 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 250 | 105 | 145 | 42.00% | 42.50% | 42.00% | 8.00 pp | -40 | 20 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 250 | 103 | 147 | 41.20% | 42.08% | 41.20% | 8.80 pp | -44 | 21 | -2.10 |
| Consolidated Hourly | nn | NN | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | transformer | Transformer | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| BTC Daily | nn | NN | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| Consolidated Market Hours Daily | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 6 | -2.67 |
| BTC Hourly | transformer | Transformer | 226 | 99 | 127 | 43.81% | 43.81% | 43.81% | 6.19 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 226 | 94 | 132 | 41.59% | 41.59% | 41.59% | 8.41 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 252 | 102 | 150 | 40.48% | 39.58% | 40.48% | 9.52 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 252 | 95 | 157 | 37.70% | 37.08% | 37.70% | 12.30 pp | -62 | 11 | -5.64 |
| BTC Hourly | lstm | LSTM | 226 | 84 | 142 | 37.17% | 37.17% | 37.17% | 12.83 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 262 | 92 | 170 | 35.11% | 35.00% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 226 | 79 | 147 | 34.96% | 34.96% | 34.96% | 15.04 pp | -68 | 10 | -6.80 |
| BTC Daily | lstm | LSTM | 252 | 87 | 165 | 34.52% | 35.42% | 34.52% | 15.48 pp | -78 | 11 | -7.09 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 252 | 102 | 150 | 40.48% | 39.58% | 40.48% | 9.52 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 252 | 95 | 157 | 37.70% | 37.08% | 37.70% | 12.30 pp | -62 | 11 | -5.64 |
| BTC Daily | xgb | XGBoost | 262 | 92 | 170 | 35.11% | 35.00% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |
| BTC Daily | lstm | LSTM | 252 | 87 | 165 | 34.52% | 35.42% | 34.52% | 15.48 pp | -78 | 11 | -7.09 |

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
| Consolidated Hourly | rf | RandomForest | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 14 | -0.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 14 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | xgb | XGBoost | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | nn | NN | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | transformer | Transformer | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 218 | 94 | 124 | 43.12% | 43.12% | 43.12% | 6.88 pp | -30 | 14 | -2.14 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
