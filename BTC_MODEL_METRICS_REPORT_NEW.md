# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T13:17:38.955459+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 292 | 232 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 328 | 268 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 12:00:00+00:00 | 478 | 256 | 222 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 12:00:00+00:00 | 478 | 256 | 222 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 120 | 138 | 46.51% | 46.25% | 46.51% | 3.49 pp | -18 | 12 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| BTC Daily | nn | NN | 258 | 117 | 141 | 45.35% | 44.17% | 45.35% | 4.65 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Hourly | transformer | Transformer | 232 | 102 | 130 | 43.97% | 43.97% | 43.97% | 6.03 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | nn | NN | 232 | 97 | 135 | 41.81% | 41.81% | 41.81% | 8.19 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 10 | -4.00 |
| BTC Daily | transformer | Transformer | 258 | 104 | 154 | 40.31% | 39.58% | 40.31% | 9.69 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 258 | 97 | 161 | 37.60% | 36.67% | 37.60% | 12.40 pp | -64 | 12 | -5.33 |
| BTC Hourly | lstm | LSTM | 232 | 87 | 145 | 37.50% | 37.50% | 37.50% | 12.50 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.00% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 258 | 90 | 168 | 34.88% | 35.42% | 34.88% | 15.12 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 232 | 80 | 152 | 34.48% | 34.48% | 34.48% | 15.52 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Hourly | transformer | Transformer | 232 | 102 | 130 | 43.97% | 43.97% | 43.97% | 6.03 pp | -28 | 10 | -2.80 |
| BTC Hourly | nn | NN | 232 | 97 | 135 | 41.81% | 41.81% | 41.81% | 8.19 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 10 | -4.00 |
| BTC Hourly | lstm | LSTM | 232 | 87 | 145 | 37.50% | 37.50% | 37.50% | 12.50 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 232 | 80 | 152 | 34.48% | 34.48% | 34.48% | 15.52 pp | -72 | 10 | -7.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 120 | 138 | 46.51% | 46.25% | 46.51% | 3.49 pp | -18 | 12 | -1.50 |
| BTC Daily | nn | NN | 258 | 117 | 141 | 45.35% | 44.17% | 45.35% | 4.65 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 258 | 104 | 154 | 40.31% | 39.58% | 40.31% | 9.69 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 258 | 97 | 161 | 37.60% | 36.67% | 37.60% | 12.40 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.00% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 258 | 90 | 168 | 34.88% | 35.42% | 34.88% | 15.12 pp | -78 | 12 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
