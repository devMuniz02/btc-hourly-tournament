# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T00:59:46.846629+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 268 | 208 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 304 | 244 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 439 | 232 | 207 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 438 | 231 | 207 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 232 | 120 | 112 | 51.72% | 51.72% | 51.72% | 1.72 pp | 8 | 18 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 231 | 114 | 117 | 49.35% | 49.35% | 49.35% | 0.65 pp | -3 | 19 | -0.16 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 9 | -0.22 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 231 | 111 | 120 | 48.05% | 48.05% | 48.05% | 1.95 pp | -9 | 19 | -0.47 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 232 | 109 | 123 | 46.98% | 46.98% | 46.98% | 3.02 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 232 | 106 | 126 | 45.69% | 45.69% | 45.69% | 4.31 pp | -20 | 18 | -1.11 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 19 | -1.32 |
| BTC Daily | mlp_sklearn | MLPClassifier | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 19 | -1.74 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| BTC Market Hours | lstm | LSTM | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 18 | -2.00 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| BTC Daily | nn | NN | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 231 | 93 | 138 | 40.26% | 40.26% | 40.26% | 9.74 pp | -45 | 19 | -2.37 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 208 | 88 | 120 | 42.31% | 42.31% | 42.31% | 7.69 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 234 | 94 | 140 | 40.17% | 40.17% | 40.17% | 9.83 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 11 | -4.91 |
| BTC Hourly | lstm | LSTM | 208 | 77 | 131 | 37.02% | 37.02% | 37.02% | 12.98 pp | -54 | 9 | -6.00 |
| BTC Daily | xgb | XGBoost | 244 | 86 | 158 | 35.25% | 35.42% | 35.25% | 14.75 pp | -72 | 12 | -6.00 |
| BTC Daily | lstm | LSTM | 234 | 78 | 156 | 33.33% | 33.33% | 33.33% | 16.67 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 208 | 71 | 137 | 34.13% | 34.13% | 34.13% | 15.87 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 9 | -0.22 |
| BTC Hourly | transformer | Transformer | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 208 | 88 | 120 | 42.31% | 42.31% | 42.31% | 7.69 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 208 | 77 | 131 | 37.02% | 37.02% | 37.02% | 12.98 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 208 | 71 | 137 | 34.13% | 34.13% | 34.13% | 15.87 pp | -66 | 9 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 11 | -2.36 |
| BTC Daily | transformer | Transformer | 234 | 94 | 140 | 40.17% | 40.17% | 40.17% | 9.83 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 244 | 86 | 158 | 35.25% | 35.42% | 35.25% | 14.75 pp | -72 | 12 | -6.00 |
| BTC Daily | lstm | LSTM | 234 | 78 | 156 | 33.33% | 33.33% | 33.33% | 16.67 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 232 | 120 | 112 | 51.72% | 51.72% | 51.72% | 1.72 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 232 | 109 | 123 | 46.98% | 46.98% | 46.98% | 3.02 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 232 | 106 | 126 | 45.69% | 45.69% | 45.69% | 4.31 pp | -20 | 18 | -1.11 |
| BTC Market Hours | lstm | LSTM | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 18 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 231 | 114 | 117 | 49.35% | 49.35% | 49.35% | 0.65 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | nn | NN | 231 | 111 | 120 | 48.05% | 48.05% | 48.05% | 1.95 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | rf | RandomForest | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 19 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 231 | 93 | 138 | 40.26% | 40.26% | 40.26% | 9.74 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
