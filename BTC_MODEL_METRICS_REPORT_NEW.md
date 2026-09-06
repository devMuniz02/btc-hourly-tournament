# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T20:26:25.909016+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 248 | 188 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 284 | 224 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 19:00:00+00:00 | 402 | 212 | 190 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 19:00:00+00:00 | 402 | 212 | 190 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 184 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | transformer | Transformer | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 18 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 188 | 95 | 93 | 50.53% | 50.53% | 50.53% | 0.53 pp | 2 | 8 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 18 | -0.22 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours | transformer | Transformer | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 17 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 212 | 101 | 111 | 47.64% | 47.64% | 47.64% | 2.36 pp | -10 | 18 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 212 | 100 | 112 | 47.17% | 47.17% | 47.17% | 2.83 pp | -12 | 17 | -0.71 |
| BTC Market Hours | rf | RandomForest | 212 | 98 | 114 | 46.23% | 46.23% | 46.23% | 3.77 pp | -16 | 17 | -0.94 |
| BTC Daily | mlp_sklearn | MLPClassifier | 214 | 102 | 112 | 47.66% | 47.66% | 47.66% | 2.34 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 212 | 94 | 118 | 44.34% | 44.34% | 44.34% | 5.66 pp | -24 | 18 | -1.33 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 212 | 92 | 120 | 43.40% | 43.40% | 43.40% | 6.60 pp | -28 | 17 | -1.65 |
| BTC Hourly | transformer | Transformer | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 212 | 87 | 125 | 41.04% | 41.04% | 41.04% | 8.96 pp | -38 | 18 | -2.11 |
| BTC Market Hours | lstm | LSTM | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 17 | -2.12 |
| BTC Daily | nn | NN | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 10 | -2.20 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 212 | 84 | 128 | 39.62% | 39.62% | 39.62% | 10.38 pp | -44 | 18 | -2.44 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 5 | -3.20 |
| BTC Hourly | nn | NN | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 214 | 89 | 125 | 41.59% | 41.59% | 41.59% | 8.41 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 8 | -3.75 |
| BTC Daily | rf | RandomForest | 214 | 82 | 132 | 38.32% | 38.32% | 38.32% | 11.68 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 224 | 81 | 143 | 36.16% | 36.16% | 36.16% | 13.84 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 188 | 70 | 118 | 37.23% | 37.23% | 37.23% | 12.77 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 188 | 69 | 119 | 36.70% | 36.70% | 36.70% | 13.30 pp | -50 | 8 | -6.25 |
| BTC Daily | lstm | LSTM | 214 | 72 | 142 | 33.64% | 33.64% | 33.64% | 16.36 pp | -70 | 10 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 188 | 95 | 93 | 50.53% | 50.53% | 50.53% | 0.53 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 8 | -1.75 |
| BTC Hourly | nn | NN | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 8 | -3.75 |
| BTC Hourly | xgb | XGBoost | 188 | 70 | 118 | 37.23% | 37.23% | 37.23% | 12.77 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 188 | 69 | 119 | 36.70% | 36.70% | 36.70% | 13.30 pp | -50 | 8 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 214 | 102 | 112 | 47.66% | 47.66% | 47.66% | 2.34 pp | -10 | 10 | -1.00 |
| BTC Daily | nn | NN | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 214 | 89 | 125 | 41.59% | 41.59% | 41.59% | 8.41 pp | -36 | 10 | -3.60 |
| BTC Daily | rf | RandomForest | 214 | 82 | 132 | 38.32% | 38.32% | 38.32% | 11.68 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 224 | 81 | 143 | 36.16% | 36.16% | 36.16% | 13.84 pp | -62 | 11 | -5.64 |
| BTC Daily | lstm | LSTM | 214 | 72 | 142 | 33.64% | 33.64% | 33.64% | 16.36 pp | -70 | 10 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 17 | 0.35 |
| BTC Market Hours | transformer | Transformer | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 17 | -0.24 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 212 | 100 | 112 | 47.17% | 47.17% | 47.17% | 2.83 pp | -12 | 17 | -0.71 |
| BTC Market Hours | rf | RandomForest | 212 | 98 | 114 | 46.23% | 46.23% | 46.23% | 3.77 pp | -16 | 17 | -0.94 |
| BTC Market Hours | xgb | XGBoost | 212 | 92 | 120 | 43.40% | 43.40% | 43.40% | 6.60 pp | -28 | 17 | -1.65 |
| BTC Market Hours | lstm | LSTM | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 17 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 18 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 18 | -0.22 |
| BTC Market Hours Daily | nn | NN | 212 | 101 | 111 | 47.64% | 47.64% | 47.64% | 2.36 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 212 | 94 | 118 | 44.34% | 44.34% | 44.34% | 5.66 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 212 | 87 | 125 | 41.04% | 41.04% | 41.04% | 8.96 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 212 | 84 | 128 | 39.62% | 39.62% | 39.62% | 10.38 pp | -44 | 18 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
