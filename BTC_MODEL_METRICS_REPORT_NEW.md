# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T21:04:59.230709+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 249 | 189 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 284 | 224 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 403 | 212 | 191 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 403 | 212 | 191 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 189 | 96 | 93 | 50.79% | 50.79% | 50.79% | 0.79 pp | 3 | 8 | 0.38 |
| BTC Market Hours | nn | NN | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | transformer | Transformer | 212 | 109 | 103 | 51.42% | 51.42% | 51.42% | 1.42 pp | 6 | 18 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 18 | -0.22 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours | transformer | Transformer | 212 | 104 | 108 | 49.06% | 49.06% | 49.06% | 0.94 pp | -4 | 17 | -0.24 |
| BTC Market Hours Daily | nn | NN | 212 | 101 | 111 | 47.64% | 47.64% | 47.64% | 2.36 pp | -10 | 18 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 212 | 100 | 112 | 47.17% | 47.17% | 47.17% | 2.83 pp | -12 | 17 | -0.71 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| BTC Market Hours | rf | RandomForest | 212 | 98 | 114 | 46.23% | 46.23% | 46.23% | 3.77 pp | -16 | 17 | -0.94 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 214 | 101 | 113 | 47.20% | 47.20% | 47.20% | 2.80 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 212 | 94 | 118 | 44.34% | 44.34% | 44.34% | 5.66 pp | -24 | 18 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 212 | 92 | 120 | 43.40% | 43.40% | 43.40% | 6.60 pp | -28 | 17 | -1.65 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Hourly | transformer | Transformer | 189 | 87 | 102 | 46.03% | 46.03% | 46.03% | 3.97 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 212 | 87 | 125 | 41.04% | 41.04% | 41.04% | 8.96 pp | -38 | 18 | -2.11 |
| BTC Market Hours | lstm | LSTM | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 17 | -2.12 |
| BTC Daily | nn | NN | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 10 | -2.20 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 212 | 84 | 128 | 39.62% | 39.62% | 39.62% | 10.38 pp | -44 | 18 | -2.44 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 8 | -3.12 |
| BTC Daily | transformer | Transformer | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 189 | 79 | 110 | 41.80% | 41.80% | 41.80% | 8.20 pp | -31 | 8 | -3.88 |
| BTC Daily | rf | RandomForest | 214 | 82 | 132 | 38.32% | 38.32% | 38.32% | 11.68 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 224 | 80 | 144 | 35.71% | 35.71% | 35.71% | 14.29 pp | -64 | 11 | -5.82 |
| BTC Hourly | lstm | LSTM | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 214 | 73 | 141 | 34.11% | 34.11% | 34.11% | 15.89 pp | -68 | 10 | -6.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 189 | 96 | 93 | 50.79% | 50.79% | 50.79% | 0.79 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 189 | 87 | 102 | 46.03% | 46.03% | 46.03% | 3.97 pp | -15 | 8 | -1.88 |
| BTC Hourly | nn | NN | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 189 | 79 | 110 | 41.80% | 41.80% | 41.80% | 8.20 pp | -31 | 8 | -3.88 |
| BTC Hourly | lstm | LSTM | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 214 | 101 | 113 | 47.20% | 47.20% | 47.20% | 2.80 pp | -12 | 10 | -1.20 |
| BTC Daily | nn | NN | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 10 | -3.80 |
| BTC Daily | rf | RandomForest | 214 | 82 | 132 | 38.32% | 38.32% | 38.32% | 11.68 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 224 | 80 | 144 | 35.71% | 35.71% | 35.71% | 14.29 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 214 | 73 | 141 | 34.11% | 34.11% | 34.11% | 15.89 pp | -68 | 10 | -6.80 |

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
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
