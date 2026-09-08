# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T07:56:16.923445+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 272 | 212 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 308 | 248 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 444 | 236 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 444 | 236 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 206 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 236 | 121 | 115 | 51.27% | 51.27% | 51.27% | 1.27 pp | 6 | 19 | 0.32 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 212 | 105 | 107 | 49.53% | 49.53% | 49.53% | 0.47 pp | -2 | 9 | -0.22 |
| BTC Market Hours Daily | nn | NN | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 19 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 19 | -0.84 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| BTC Market Hours | rf | RandomForest | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | rf | RandomForest | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 20 | -1.30 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 236 | 103 | 133 | 43.64% | 43.64% | 43.64% | 6.36 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 236 | 101 | 135 | 42.80% | 42.80% | 42.80% | 7.20 pp | -34 | 19 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 11 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 20 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| BTC Hourly | transformer | Transformer | 212 | 95 | 117 | 44.81% | 44.81% | 44.81% | 5.19 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Hourly | nn | NN | 212 | 89 | 123 | 41.98% | 41.98% | 41.98% | 8.02 pp | -34 | 9 | -3.78 |
| BTC Hourly | rf | RandomForest | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 238 | 96 | 142 | 40.34% | 40.34% | 40.34% | 9.66 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 238 | 92 | 146 | 38.66% | 38.66% | 38.66% | 11.34 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 248 | 89 | 159 | 35.89% | 36.25% | 35.89% | 14.11 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 212 | 79 | 133 | 37.26% | 37.26% | 37.26% | 12.74 pp | -54 | 9 | -6.00 |
| BTC Daily | lstm | LSTM | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 11 | -6.91 |
| BTC Hourly | xgb | XGBoost | 212 | 73 | 139 | 34.43% | 34.43% | 34.43% | 15.57 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 212 | 105 | 107 | 49.53% | 49.53% | 49.53% | 0.47 pp | -2 | 9 | -0.22 |
| BTC Hourly | transformer | Transformer | 212 | 95 | 117 | 44.81% | 44.81% | 44.81% | 5.19 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 212 | 89 | 123 | 41.98% | 41.98% | 41.98% | 8.02 pp | -34 | 9 | -3.78 |
| BTC Hourly | rf | RandomForest | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 212 | 79 | 133 | 37.26% | 37.26% | 37.26% | 12.74 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 212 | 73 | 139 | 34.43% | 34.43% | 34.43% | 15.57 pp | -66 | 9 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 11 | -1.27 |
| BTC Daily | nn | NN | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 238 | 96 | 142 | 40.34% | 40.34% | 40.34% | 9.66 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 238 | 92 | 146 | 38.66% | 38.66% | 38.66% | 11.34 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 248 | 89 | 159 | 35.89% | 36.25% | 35.89% | 14.11 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 11 | -6.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 236 | 121 | 115 | 51.27% | 51.27% | 51.27% | 1.27 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | lstm | LSTM | 236 | 101 | 135 | 42.80% | 42.80% | 42.80% | 7.20 pp | -34 | 19 | -1.79 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | nn | NN | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | rf | RandomForest | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 236 | 103 | 133 | 43.64% | 43.64% | 43.64% | 6.36 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 20 | -2.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
