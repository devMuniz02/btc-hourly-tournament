# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T10:51:56.208378+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 193 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 222 | 116 | 106 | 52.25% | 52.25% | 52.25% | 2.25 pp | 10 | 18 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 198 | 99 | 99 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | nn | NN | 222 | 108 | 114 | 48.65% | 48.65% | 48.65% | 1.35 pp | -6 | 18 | -0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | transformer | Transformer | 222 | 107 | 115 | 48.20% | 48.20% | 48.20% | 1.80 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours | rf | RandomForest | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 222 | 102 | 120 | 45.95% | 45.95% | 45.95% | 4.05 pp | -18 | 18 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 222 | 101 | 121 | 45.50% | 45.50% | 45.50% | 4.50 pp | -20 | 18 | -1.11 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 18 | -1.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 10 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | xgb | XGBoost | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 198 | 88 | 110 | 44.44% | 44.44% | 44.44% | 5.56 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Market Hours | lstm | LSTM | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 18 | -3.78 |
| BTC Market Hours Daily | lstm | LSTM | 222 | 76 | 146 | 34.23% | 34.23% | 34.23% | 15.77 pp | -70 | 18 | -3.89 |
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
| BTC Market Hours | nn | NN | 222 | 116 | 106 | 52.25% | 52.25% | 52.25% | 2.25 pp | 10 | 18 | 0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours | rf | RandomForest | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 222 | 101 | 121 | 45.50% | 45.50% | 45.50% | 4.50 pp | -20 | 18 | -1.11 |
| BTC Market Hours | xgb | XGBoost | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 18 | -1.78 |
| BTC Market Hours | lstm | LSTM | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 18 | -3.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 222 | 108 | 114 | 48.65% | 48.65% | 48.65% | 1.35 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 222 | 107 | 115 | 48.20% | 48.20% | 48.20% | 1.80 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 222 | 102 | 120 | 45.95% | 45.95% | 45.95% | 4.05 pp | -18 | 18 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 222 | 76 | 146 | 34.23% | 34.23% | 34.23% | 15.77 pp | -70 | 18 | -3.89 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Hourly | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
