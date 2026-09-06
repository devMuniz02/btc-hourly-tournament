# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T23:30:12.874363+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 250 | 190 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 286 | 226 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 22:00:00+00:00 | 407 | 214 | 193 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 22:00:00+00:00 | 407 | 214 | 193 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 214 | 110 | 104 | 51.40% | 51.40% | 51.40% | 1.40 pp | 6 | 18 | 0.33 |
| BTC Market Hours | nn | NN | 214 | 109 | 105 | 50.93% | 50.93% | 50.93% | 0.93 pp | 4 | 17 | 0.24 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 214 | 104 | 110 | 48.60% | 48.60% | 48.60% | 1.40 pp | -6 | 18 | -0.33 |
| BTC Market Hours | transformer | Transformer | 214 | 104 | 110 | 48.60% | 48.60% | 48.60% | 1.40 pp | -6 | 17 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 214 | 101 | 113 | 47.20% | 47.20% | 47.20% | 2.80 pp | -12 | 18 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 214 | 100 | 114 | 46.73% | 46.73% | 46.73% | 3.27 pp | -14 | 17 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| BTC Market Hours | rf | RandomForest | 214 | 98 | 116 | 45.79% | 45.79% | 45.79% | 4.21 pp | -18 | 17 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| BTC Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 10 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 214 | 95 | 119 | 44.39% | 44.39% | 44.39% | 5.61 pp | -24 | 18 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 214 | 93 | 121 | 43.46% | 43.46% | 43.46% | 6.54 pp | -28 | 17 | -1.65 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| BTC Hourly | transformer | Transformer | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 8 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 18 | -2.11 |
| BTC Daily | nn | NN | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 10 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Market Hours | lstm | LSTM | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 17 | -2.24 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 214 | 85 | 129 | 39.72% | 39.72% | 39.72% | 10.28 pp | -44 | 18 | -2.44 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 190 | 79 | 111 | 41.58% | 41.58% | 41.58% | 8.42 pp | -32 | 8 | -4.00 |
| BTC Daily | rf | RandomForest | 216 | 83 | 133 | 38.43% | 38.43% | 38.43% | 11.57 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 226 | 81 | 145 | 35.84% | 35.84% | 35.84% | 14.16 pp | -64 | 11 | -5.82 |
| BTC Hourly | lstm | LSTM | 190 | 71 | 119 | 37.37% | 37.37% | 37.37% | 12.63 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 190 | 70 | 120 | 36.84% | 36.84% | 36.84% | 13.16 pp | -50 | 8 | -6.25 |
| BTC Daily | lstm | LSTM | 216 | 73 | 143 | 33.80% | 33.80% | 33.80% | 16.20 pp | -70 | 10 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 8 | -2.00 |
| BTC Hourly | nn | NN | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 190 | 79 | 111 | 41.58% | 41.58% | 41.58% | 8.42 pp | -32 | 8 | -4.00 |
| BTC Hourly | lstm | LSTM | 190 | 71 | 119 | 37.37% | 37.37% | 37.37% | 12.63 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 190 | 70 | 120 | 36.84% | 36.84% | 36.84% | 13.16 pp | -50 | 8 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 10 | -1.20 |
| BTC Daily | nn | NN | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 216 | 89 | 127 | 41.20% | 41.20% | 41.20% | 8.80 pp | -38 | 10 | -3.80 |
| BTC Daily | rf | RandomForest | 216 | 83 | 133 | 38.43% | 38.43% | 38.43% | 11.57 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 226 | 81 | 145 | 35.84% | 35.84% | 35.84% | 14.16 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 216 | 73 | 143 | 33.80% | 33.80% | 33.80% | 16.20 pp | -70 | 10 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 214 | 109 | 105 | 50.93% | 50.93% | 50.93% | 0.93 pp | 4 | 17 | 0.24 |
| BTC Market Hours | transformer | Transformer | 214 | 104 | 110 | 48.60% | 48.60% | 48.60% | 1.40 pp | -6 | 17 | -0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 214 | 100 | 114 | 46.73% | 46.73% | 46.73% | 3.27 pp | -14 | 17 | -0.82 |
| BTC Market Hours | rf | RandomForest | 214 | 98 | 116 | 45.79% | 45.79% | 45.79% | 4.21 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 214 | 93 | 121 | 43.46% | 43.46% | 43.46% | 6.54 pp | -28 | 17 | -1.65 |
| BTC Market Hours | lstm | LSTM | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 17 | -2.24 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 214 | 110 | 104 | 51.40% | 51.40% | 51.40% | 1.40 pp | 6 | 18 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 214 | 104 | 110 | 48.60% | 48.60% | 48.60% | 1.40 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | nn | NN | 214 | 101 | 113 | 47.20% | 47.20% | 47.20% | 2.80 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 214 | 95 | 119 | 44.39% | 44.39% | 44.39% | 5.61 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 214 | 85 | 129 | 39.72% | 39.72% | 39.72% | 10.28 pp | -44 | 18 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
