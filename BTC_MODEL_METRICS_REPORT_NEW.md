# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T22:07:59.726471+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 266 | 206 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 302 | 242 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 21:00:00+00:00 | 435 | 230 | 205 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 21:00:00+00:00 | 434 | 229 | 205 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 230 | 119 | 111 | 51.74% | 51.74% | 51.74% | 1.74 pp | 8 | 18 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 206 | 103 | 103 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 19 | -0.26 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | nn | NN | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 19 | -0.58 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 230 | 109 | 121 | 47.39% | 47.39% | 47.39% | 2.61 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 230 | 104 | 126 | 45.22% | 45.22% | 45.22% | 4.78 pp | -22 | 18 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 19 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 232 | 108 | 124 | 46.55% | 46.55% | 46.55% | 3.45 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 19 | -1.84 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 19 | -2.37 |
| BTC Daily | nn | NN | 232 | 104 | 128 | 44.83% | 44.83% | 44.83% | 5.17 pp | -24 | 10 | -2.40 |
| BTC Hourly | transformer | Transformer | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 206 | 87 | 119 | 42.23% | 42.23% | 42.23% | 7.77 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 206 | 85 | 121 | 41.26% | 41.26% | 41.26% | 8.74 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 232 | 89 | 143 | 38.36% | 38.36% | 38.36% | 11.64 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 206 | 76 | 130 | 36.89% | 36.89% | 36.89% | 13.11 pp | -54 | 9 | -6.00 |
| BTC Daily | xgb | XGBoost | 242 | 86 | 156 | 35.54% | 35.42% | 35.54% | 14.46 pp | -70 | 11 | -6.36 |
| BTC Hourly | xgb | XGBoost | 206 | 71 | 135 | 34.47% | 34.47% | 34.47% | 15.53 pp | -64 | 9 | -7.11 |
| BTC Daily | lstm | LSTM | 232 | 77 | 155 | 33.19% | 33.19% | 33.19% | 16.81 pp | -78 | 10 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 206 | 103 | 103 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 206 | 87 | 119 | 42.23% | 42.23% | 42.23% | 7.77 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 206 | 85 | 121 | 41.26% | 41.26% | 41.26% | 8.74 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 206 | 76 | 130 | 36.89% | 36.89% | 36.89% | 13.11 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 206 | 71 | 135 | 34.47% | 34.47% | 34.47% | 15.53 pp | -64 | 9 | -7.11 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 232 | 108 | 124 | 46.55% | 46.55% | 46.55% | 3.45 pp | -16 | 10 | -1.60 |
| BTC Daily | nn | NN | 232 | 104 | 128 | 44.83% | 44.83% | 44.83% | 5.17 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 232 | 89 | 143 | 38.36% | 38.36% | 38.36% | 11.64 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 242 | 86 | 156 | 35.54% | 35.42% | 35.54% | 14.46 pp | -70 | 11 | -6.36 |
| BTC Daily | lstm | LSTM | 232 | 77 | 155 | 33.19% | 33.19% | 33.19% | 16.81 pp | -78 | 10 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 230 | 119 | 111 | 51.74% | 51.74% | 51.74% | 1.74 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 230 | 109 | 121 | 47.39% | 47.39% | 47.39% | 2.61 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 230 | 104 | 126 | 45.22% | 45.22% | 45.22% | 4.78 pp | -22 | 18 | -1.22 |
| BTC Market Hours | lstm | LSTM | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 18 | -2.11 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 19 | -0.26 |
| BTC Market Hours Daily | nn | NN | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 19 | -0.58 |
| BTC Market Hours Daily | rf | RandomForest | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 19 | -1.84 |
| BTC Market Hours Daily | lstm | LSTM | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
