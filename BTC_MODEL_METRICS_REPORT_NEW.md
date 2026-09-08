# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T05:04:43.853610+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 270 | 210 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 306 | 246 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 442 | 234 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 442 | 234 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 204 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 234 | 121 | 113 | 51.71% | 51.71% | 51.71% | 1.71 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 210 | 105 | 105 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | transformer | Transformer | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 19 | -0.21 |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| BTC Market Hours Daily | nn | NN | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 19 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 18 | -0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 18 | -1.11 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 19 | -1.37 |
| BTC Daily | mlp_sklearn | MLPClassifier | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 11 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 234 | 102 | 132 | 43.59% | 43.59% | 43.59% | 6.41 pp | -30 | 19 | -1.58 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 18 | -2.00 |
| BTC Daily | nn | NN | 236 | 106 | 130 | 44.92% | 44.92% | 44.92% | 5.08 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 234 | 95 | 139 | 40.60% | 40.60% | 40.60% | 9.40 pp | -44 | 19 | -2.32 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| BTC Hourly | transformer | Transformer | 210 | 94 | 116 | 44.76% | 44.76% | 44.76% | 5.24 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Hourly | nn | NN | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 11 | -4.00 |
| BTC Daily | rf | RandomForest | 236 | 91 | 145 | 38.56% | 38.56% | 38.56% | 11.44 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 246 | 88 | 158 | 35.77% | 35.83% | 35.77% | 14.23 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 210 | 78 | 132 | 37.14% | 37.14% | 37.14% | 12.86 pp | -54 | 9 | -6.00 |
| BTC Daily | lstm | LSTM | 236 | 79 | 157 | 33.47% | 33.47% | 33.47% | 16.53 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 210 | 72 | 138 | 34.29% | 34.29% | 34.29% | 15.71 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 210 | 105 | 105 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 210 | 94 | 116 | 44.76% | 44.76% | 44.76% | 5.24 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 210 | 78 | 132 | 37.14% | 37.14% | 37.14% | 12.86 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 210 | 72 | 138 | 34.29% | 34.29% | 34.29% | 15.71 pp | -66 | 9 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 236 | 106 | 130 | 44.92% | 44.92% | 44.92% | 5.08 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 11 | -4.00 |
| BTC Daily | rf | RandomForest | 236 | 91 | 145 | 38.56% | 38.56% | 38.56% | 11.44 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 246 | 88 | 158 | 35.77% | 35.83% | 35.77% | 14.23 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 236 | 79 | 157 | 33.47% | 33.47% | 33.47% | 16.53 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 234 | 121 | 113 | 51.71% | 51.71% | 51.71% | 1.71 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 18 | -1.11 |
| BTC Market Hours | lstm | LSTM | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 18 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | transformer | Transformer | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | nn | NN | 234 | 112 | 122 | 47.86% | 47.86% | 47.86% | 2.14 pp | -10 | 19 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 19 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 234 | 102 | 132 | 43.59% | 43.59% | 43.59% | 6.41 pp | -30 | 19 | -1.58 |
| BTC Market Hours Daily | lstm | LSTM | 234 | 95 | 139 | 40.60% | 40.60% | 40.60% | 9.40 pp | -44 | 19 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
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
