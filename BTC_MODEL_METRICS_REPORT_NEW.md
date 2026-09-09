# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T22:08:50.334757+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 297 | 237 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 333 | 273 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 492 | 261 | 231 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 492 | 261 | 231 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 228 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 261 | 137 | 124 | 52.49% | 52.50% | 52.49% | 2.49 pp | 13 | 21 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 261 | 130 | 131 | 49.81% | 49.58% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 261 | 128 | 133 | 49.04% | 48.75% | 49.04% | 0.96 pp | -5 | 21 | -0.24 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 10 | -0.50 |
| BTC Market Hours Daily | nn | NN | 261 | 125 | 136 | 47.89% | 48.33% | 47.89% | 2.11 pp | -11 | 21 | -0.52 |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 261 | 123 | 138 | 47.13% | 47.92% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 261 | 123 | 138 | 47.13% | 47.50% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | xgb | XGBoost | 261 | 120 | 141 | 45.98% | 44.17% | 45.98% | 4.02 pp | -21 | 21 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 261 | 116 | 145 | 44.44% | 43.33% | 44.44% | 5.56 pp | -29 | 21 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 261 | 115 | 146 | 44.06% | 42.92% | 44.06% | 5.94 pp | -31 | 21 | -1.48 |
| BTC Daily | mlp_sklearn | MLPClassifier | 263 | 122 | 141 | 46.39% | 45.42% | 46.39% | 3.61 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 261 | 113 | 148 | 43.30% | 42.50% | 43.30% | 6.70 pp | -35 | 21 | -1.67 |
| BTC Daily | nn | NN | 263 | 121 | 142 | 46.01% | 45.00% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 261 | 108 | 153 | 41.38% | 42.08% | 41.38% | 8.62 pp | -45 | 21 | -2.14 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| BTC Market Hours | lstm | LSTM | 261 | 106 | 155 | 40.61% | 42.08% | 40.61% | 9.39 pp | -49 | 21 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 237 | 100 | 137 | 42.19% | 42.19% | 42.19% | 7.81 pp | -37 | 10 | -3.70 |
| BTC Daily | transformer | Transformer | 263 | 107 | 156 | 40.68% | 40.00% | 40.68% | 9.32 pp | -49 | 12 | -4.08 |
| BTC Hourly | rf | RandomForest | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 263 | 99 | 164 | 37.64% | 37.50% | 37.64% | 12.36 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 273 | 100 | 173 | 36.63% | 36.67% | 36.63% | 13.37 pp | -73 | 13 | -5.62 |
| BTC Hourly | lstm | LSTM | 237 | 87 | 150 | 36.71% | 36.71% | 36.71% | 13.29 pp | -63 | 10 | -6.30 |
| BTC Daily | lstm | LSTM | 263 | 93 | 170 | 35.36% | 36.25% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 237 | 81 | 156 | 34.18% | 34.18% | 34.18% | 15.82 pp | -75 | 10 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 10 | -0.50 |
| BTC Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 237 | 100 | 137 | 42.19% | 42.19% | 42.19% | 7.81 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 10 | -4.10 |
| BTC Hourly | lstm | LSTM | 237 | 87 | 150 | 36.71% | 36.71% | 36.71% | 13.29 pp | -63 | 10 | -6.30 |
| BTC Hourly | xgb | XGBoost | 237 | 81 | 156 | 34.18% | 34.18% | 34.18% | 15.82 pp | -75 | 10 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 263 | 122 | 141 | 46.39% | 45.42% | 46.39% | 3.61 pp | -19 | 12 | -1.58 |
| BTC Daily | nn | NN | 263 | 121 | 142 | 46.01% | 45.00% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 263 | 107 | 156 | 40.68% | 40.00% | 40.68% | 9.32 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 263 | 99 | 164 | 37.64% | 37.50% | 37.64% | 12.36 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 273 | 100 | 173 | 36.63% | 36.67% | 36.63% | 13.37 pp | -73 | 13 | -5.62 |
| BTC Daily | lstm | LSTM | 263 | 93 | 170 | 35.36% | 36.25% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 261 | 137 | 124 | 52.49% | 52.50% | 52.49% | 2.49 pp | 13 | 21 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 261 | 123 | 138 | 47.13% | 47.92% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 261 | 123 | 138 | 47.13% | 47.50% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | xgb | XGBoost | 261 | 120 | 141 | 45.98% | 44.17% | 45.98% | 4.02 pp | -21 | 21 | -1.00 |
| BTC Market Hours | rf | RandomForest | 261 | 116 | 145 | 44.44% | 43.33% | 44.44% | 5.56 pp | -29 | 21 | -1.38 |
| BTC Market Hours | lstm | LSTM | 261 | 106 | 155 | 40.61% | 42.08% | 40.61% | 9.39 pp | -49 | 21 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 261 | 130 | 131 | 49.81% | 49.58% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 261 | 128 | 133 | 49.04% | 48.75% | 49.04% | 0.96 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | nn | NN | 261 | 125 | 136 | 47.89% | 48.33% | 47.89% | 2.11 pp | -11 | 21 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 261 | 115 | 146 | 44.06% | 42.92% | 44.06% | 5.94 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 261 | 113 | 148 | 43.30% | 42.50% | 43.30% | 6.70 pp | -35 | 21 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 261 | 108 | 153 | 41.38% | 42.08% | 41.38% | 8.62 pp | -45 | 21 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
