# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T10:20:37.937522+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 416 | 221 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 191 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 191 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 59 | 132 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 59 | 132 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 222 | 116 | 106 | 52.25% | 52.25% | 52.25% | 2.25 pp | 10 | 18 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 198 | 99 | 99 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 221 | 108 | 113 | 48.87% | 48.87% | 48.87% | 1.13 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 18 | -0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours | rf | RandomForest | 222 | 106 | 116 | 47.75% | 47.75% | 47.75% | 2.25 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 18 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| BTC Market Hours | transformer | Transformer | 222 | 101 | 121 | 45.50% | 45.50% | 45.50% | 4.50 pp | -20 | 18 | -1.11 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 18 | -1.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 10 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 18 | -1.94 |
| Consolidated Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 10 | -2.40 |
| BTC Hourly | transformer | Transformer | 198 | 88 | 110 | 44.44% | 44.44% | 44.44% | 5.56 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 198 | 83 | 115 | 41.92% | 41.92% | 41.92% | 8.08 pp | -32 | 9 | -3.56 |
| BTC Market Hours | lstm | LSTM | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 18 | -3.78 |
| BTC Market Hours Daily | lstm | LSTM | 221 | 76 | 145 | 34.39% | 34.39% | 34.39% | 15.61 pp | -69 | 18 | -3.83 |
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
| BTC Market Hours Daily | nn | NN | 221 | 108 | 113 | 48.87% | 48.87% | 48.87% | 1.13 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 18 | -0.39 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 18 | -0.94 |
| BTC Market Hours Daily | xgb | XGBoost | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 18 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 221 | 76 | 145 | 34.39% | 34.39% | 34.39% | 15.61 pp | -69 | 18 | -3.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
