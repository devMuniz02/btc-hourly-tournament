# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T09:56:29.266816+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 273 | 213 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 309 | 249 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 445 | 237 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 445 | 237 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 237 | 121 | 116 | 51.05% | 51.05% | 51.05% | 1.05 pp | 5 | 19 | 0.26 |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 20 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 213 | 105 | 108 | 49.30% | 49.30% | 49.30% | 0.70 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | nn | NN | 237 | 112 | 125 | 47.26% | 47.26% | 47.26% | 2.74 pp | -13 | 20 | -0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 237 | 112 | 125 | 47.26% | 47.26% | 47.26% | 2.74 pp | -13 | 19 | -0.68 |
| BTC Market Hours | transformer | Transformer | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 19 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | xgb | XGBoost | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 19 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 237 | 108 | 129 | 45.57% | 45.57% | 45.57% | 4.43 pp | -21 | 19 | -1.11 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 20 | -1.35 |
| BTC Daily | mlp_sklearn | MLPClassifier | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 20 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 237 | 101 | 136 | 42.62% | 42.62% | 42.62% | 7.38 pp | -35 | 19 | -1.84 |
| BTC Daily | nn | NN | 239 | 109 | 130 | 45.61% | 45.61% | 45.61% | 4.39 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 237 | 97 | 140 | 40.93% | 40.93% | 40.93% | 9.07 pp | -43 | 20 | -2.15 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Hourly | transformer | Transformer | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Hourly | nn | NN | 213 | 90 | 123 | 42.25% | 42.25% | 42.25% | 7.75 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 239 | 97 | 142 | 40.59% | 40.59% | 40.59% | 9.41 pp | -45 | 11 | -4.09 |
| BTC Hourly | rf | RandomForest | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 9 | -4.11 |
| BTC Daily | rf | RandomForest | 239 | 92 | 147 | 38.49% | 38.49% | 38.49% | 11.51 pp | -55 | 11 | -5.00 |
| BTC Daily | xgb | XGBoost | 249 | 90 | 159 | 36.14% | 36.25% | 36.14% | 13.86 pp | -69 | 12 | -5.75 |
| BTC Hourly | lstm | LSTM | 213 | 79 | 134 | 37.09% | 37.09% | 37.09% | 12.91 pp | -55 | 9 | -6.11 |
| BTC Daily | lstm | LSTM | 239 | 81 | 158 | 33.89% | 33.89% | 33.89% | 16.11 pp | -77 | 11 | -7.00 |
| BTC Hourly | xgb | XGBoost | 213 | 73 | 140 | 34.27% | 34.27% | 34.27% | 15.73 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 213 | 105 | 108 | 49.30% | 49.30% | 49.30% | 0.70 pp | -3 | 9 | -0.33 |
| BTC Hourly | transformer | Transformer | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 213 | 90 | 123 | 42.25% | 42.25% | 42.25% | 7.75 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 213 | 79 | 134 | 37.09% | 37.09% | 37.09% | 12.91 pp | -55 | 9 | -6.11 |
| BTC Hourly | xgb | XGBoost | 213 | 73 | 140 | 34.27% | 34.27% | 34.27% | 15.73 pp | -67 | 9 | -7.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 239 | 109 | 130 | 45.61% | 45.61% | 45.61% | 4.39 pp | -21 | 11 | -1.91 |
| BTC Daily | transformer | Transformer | 239 | 97 | 142 | 40.59% | 40.59% | 40.59% | 9.41 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 239 | 92 | 147 | 38.49% | 38.49% | 38.49% | 11.51 pp | -55 | 11 | -5.00 |
| BTC Daily | xgb | XGBoost | 249 | 90 | 159 | 36.14% | 36.25% | 36.14% | 13.86 pp | -69 | 12 | -5.75 |
| BTC Daily | lstm | LSTM | 239 | 81 | 158 | 33.89% | 33.89% | 33.89% | 16.11 pp | -77 | 11 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 237 | 121 | 116 | 51.05% | 51.05% | 51.05% | 1.05 pp | 5 | 19 | 0.26 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 237 | 112 | 125 | 47.26% | 47.26% | 47.26% | 2.74 pp | -13 | 19 | -0.68 |
| BTC Market Hours | transformer | Transformer | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 19 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 19 | -1.00 |
| BTC Market Hours | rf | RandomForest | 237 | 108 | 129 | 45.57% | 45.57% | 45.57% | 4.43 pp | -21 | 19 | -1.11 |
| BTC Market Hours | lstm | LSTM | 237 | 101 | 136 | 42.62% | 42.62% | 42.62% | 7.38 pp | -35 | 19 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 237 | 116 | 121 | 48.95% | 48.95% | 48.95% | 1.05 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | nn | NN | 237 | 112 | 125 | 47.26% | 47.26% | 47.26% | 2.74 pp | -13 | 20 | -0.65 |
| BTC Market Hours Daily | rf | RandomForest | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 20 | -1.55 |
| BTC Market Hours Daily | lstm | LSTM | 237 | 97 | 140 | 40.93% | 40.93% | 40.93% | 9.07 pp | -43 | 20 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
