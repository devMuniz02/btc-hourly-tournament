# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T14:11:14.189715+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 276 | 216 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 312 | 252 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 450 | 240 | 210 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 450 | 240 | 210 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 210 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 240 | 122 | 118 | 50.83% | 50.83% | 50.83% | 0.83 pp | 4 | 19 | 0.21 |
| BTC Market Hours Daily | transformer | Transformer | 240 | 118 | 122 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 240 | 116 | 124 | 48.33% | 48.33% | 48.33% | 1.67 pp | -8 | 20 | -0.40 |
| Consolidated Hourly | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 9 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | nn | NN | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 20 | -0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 19 | -0.74 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 19 | -1.16 |
| Consolidated Hourly | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 242 | 114 | 128 | 47.11% | 46.67% | 47.11% | 2.89 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 240 | 106 | 134 | 44.17% | 44.17% | 44.17% | 5.83 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 242 | 110 | 132 | 45.45% | 45.00% | 45.45% | 4.55 pp | -22 | 11 | -2.00 |
| BTC Market Hours | lstm | LSTM | 240 | 101 | 139 | 42.08% | 42.08% | 42.08% | 7.92 pp | -38 | 19 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 240 | 97 | 143 | 40.42% | 40.42% | 40.42% | 9.58 pp | -46 | 20 | -2.30 |
| BTC Hourly | transformer | Transformer | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 216 | 91 | 125 | 42.13% | 42.13% | 42.13% | 7.87 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 242 | 98 | 144 | 40.50% | 40.00% | 40.50% | 9.50 pp | -46 | 11 | -4.18 |
| BTC Hourly | rf | RandomForest | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 9 | -4.22 |
| BTC Daily | rf | RandomForest | 242 | 92 | 150 | 38.02% | 37.50% | 38.02% | 11.98 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 252 | 90 | 162 | 35.71% | 35.42% | 35.71% | 14.29 pp | -72 | 12 | -6.00 |
| BTC Hourly | lstm | LSTM | 216 | 80 | 136 | 37.04% | 37.04% | 37.04% | 12.96 pp | -56 | 9 | -6.22 |
| BTC Daily | lstm | LSTM | 242 | 81 | 161 | 33.47% | 33.75% | 33.47% | 16.53 pp | -80 | 11 | -7.27 |
| BTC Hourly | xgb | XGBoost | 216 | 74 | 142 | 34.26% | 34.26% | 34.26% | 15.74 pp | -68 | 9 | -7.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 9 | -0.67 |
| BTC Hourly | transformer | Transformer | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 216 | 91 | 125 | 42.13% | 42.13% | 42.13% | 7.87 pp | -34 | 9 | -3.78 |
| BTC Hourly | rf | RandomForest | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 9 | -4.22 |
| BTC Hourly | lstm | LSTM | 216 | 80 | 136 | 37.04% | 37.04% | 37.04% | 12.96 pp | -56 | 9 | -6.22 |
| BTC Hourly | xgb | XGBoost | 216 | 74 | 142 | 34.26% | 34.26% | 34.26% | 15.74 pp | -68 | 9 | -7.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 242 | 114 | 128 | 47.11% | 46.67% | 47.11% | 2.89 pp | -14 | 11 | -1.27 |
| BTC Daily | nn | NN | 242 | 110 | 132 | 45.45% | 45.00% | 45.45% | 4.55 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 242 | 98 | 144 | 40.50% | 40.00% | 40.50% | 9.50 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 242 | 92 | 150 | 38.02% | 37.50% | 38.02% | 11.98 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 252 | 90 | 162 | 35.71% | 35.42% | 35.71% | 14.29 pp | -72 | 12 | -6.00 |
| BTC Daily | lstm | LSTM | 242 | 81 | 161 | 33.47% | 33.75% | 33.47% | 16.53 pp | -80 | 11 | -7.27 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 240 | 122 | 118 | 50.83% | 50.83% | 50.83% | 0.83 pp | 4 | 19 | 0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 19 | -0.74 |
| BTC Market Hours | transformer | Transformer | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 19 | -1.16 |
| BTC Market Hours | lstm | LSTM | 240 | 101 | 139 | 42.08% | 42.08% | 42.08% | 7.92 pp | -38 | 19 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 240 | 118 | 122 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 240 | 116 | 124 | 48.33% | 48.33% | 48.33% | 1.67 pp | -8 | 20 | -0.40 |
| BTC Market Hours Daily | nn | NN | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 20 | -0.70 |
| BTC Market Hours Daily | xgb | XGBoost | 240 | 106 | 134 | 44.17% | 44.17% | 44.17% | 5.83 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 240 | 97 | 143 | 40.42% | 40.42% | 40.42% | 9.58 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
