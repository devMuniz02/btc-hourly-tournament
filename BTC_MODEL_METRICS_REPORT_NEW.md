# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T11:03:55.600198+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 258 | 198 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 294 | 234 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 417 | 222 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 417 | 222 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 222 | 115 | 107 | 51.80% | 51.80% | 51.80% | 1.80 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | transformer | Transformer | 222 | 112 | 110 | 50.45% | 50.45% | 50.45% | 0.45 pp | 2 | 18 | 0.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 198 | 99 | 99 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| BTC Market Hours Daily | nn | NN | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 222 | 99 | 123 | 44.59% | 44.59% | 44.59% | 5.41 pp | -24 | 18 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 18 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 10 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 18 | -2.11 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Market Hours | lstm | LSTM | 222 | 90 | 132 | 40.54% | 40.54% | 40.54% | 9.46 pp | -42 | 18 | -2.33 |
| BTC Daily | nn | NN | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 198 | 88 | 110 | 44.44% | 44.44% | 44.44% | 5.56 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 222 | 87 | 135 | 39.19% | 39.19% | 39.19% | 10.81 pp | -48 | 18 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Daily | transformer | Transformer | 224 | 91 | 133 | 40.62% | 40.62% | 40.62% | 9.38 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 224 | 86 | 138 | 38.39% | 38.39% | 38.39% | 11.61 pp | -52 | 10 | -5.20 |
| BTC Hourly | lstm | LSTM | 198 | 74 | 124 | 37.37% | 37.37% | 37.37% | 12.63 pp | -50 | 9 | -5.56 |
| BTC Daily | xgb | XGBoost | 234 | 83 | 151 | 35.47% | 35.47% | 35.47% | 14.53 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 198 | 71 | 127 | 35.86% | 35.86% | 35.86% | 14.14 pp | -56 | 9 | -6.22 |
| BTC Daily | lstm | LSTM | 224 | 75 | 149 | 33.48% | 33.48% | 33.48% | 16.52 pp | -74 | 10 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 198 | 99 | 99 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 198 | 88 | 110 | 44.44% | 44.44% | 44.44% | 5.56 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Hourly | lstm | LSTM | 198 | 74 | 124 | 37.37% | 37.37% | 37.37% | 12.63 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 198 | 71 | 127 | 35.86% | 35.86% | 35.86% | 14.14 pp | -56 | 9 | -6.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 10 | -1.80 |
| BTC Daily | nn | NN | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 224 | 91 | 133 | 40.62% | 40.62% | 40.62% | 9.38 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 224 | 86 | 138 | 38.39% | 38.39% | 38.39% | 11.61 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 234 | 83 | 151 | 35.47% | 35.47% | 35.47% | 14.53 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 224 | 75 | 149 | 33.48% | 33.48% | 33.48% | 16.52 pp | -74 | 10 | -7.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 222 | 115 | 107 | 51.80% | 51.80% | 51.80% | 1.80 pp | 8 | 18 | 0.44 |
| BTC Market Hours | transformer | Transformer | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| BTC Market Hours | xgb | XGBoost | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 18 | -1.44 |
| BTC Market Hours | lstm | LSTM | 222 | 90 | 132 | 40.54% | 40.54% | 40.54% | 9.46 pp | -42 | 18 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 222 | 112 | 110 | 50.45% | 50.45% | 50.45% | 0.45 pp | 2 | 18 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | nn | NN | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 18 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 222 | 99 | 123 | 44.59% | 44.59% | 44.59% | 5.41 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 222 | 87 | 135 | 39.19% | 39.19% | 39.19% | 10.81 pp | -48 | 18 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
