# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T22:38:09.369182+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 21:00:00+00:00 | 435 | 230 | 205 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 200 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 200 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 200 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 201 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 230 | 119 | 111 | 51.74% | 51.74% | 51.74% | 1.74 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 206 | 103 | 103 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 230 | 114 | 116 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 230 | 113 | 117 | 49.13% | 49.13% | 49.13% | 0.87 pp | -4 | 19 | -0.21 |
| Consolidated Hourly | rf | RandomForest | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | nn | NN | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 19 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 18 | -0.56 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 230 | 109 | 121 | 47.39% | 47.39% | 47.39% | 2.61 pp | -12 | 18 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| BTC Market Hours | rf | RandomForest | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 230 | 104 | 126 | 45.22% | 45.22% | 45.22% | 4.78 pp | -22 | 18 | -1.22 |
| Consolidated Hourly | xgb | XGBoost | 200 | 92 | 108 | 46.00% | 46.00% | 46.00% | 4.00 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 200 | 92 | 108 | 46.00% | 46.00% | 46.00% | 4.00 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 19 | -1.26 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | lstm | LSTM | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 232 | 108 | 124 | 46.55% | 46.55% | 46.55% | 3.45 pp | -16 | 10 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 19 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 230 | 93 | 137 | 40.43% | 40.43% | 40.43% | 9.57 pp | -44 | 19 | -2.32 |
| BTC Daily | nn | NN | 232 | 104 | 128 | 44.83% | 44.83% | 44.83% | 5.17 pp | -24 | 10 | -2.40 |
| BTC Hourly | transformer | Transformer | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
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
| BTC Market Hours Daily | transformer | Transformer | 230 | 114 | 116 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 230 | 113 | 117 | 49.13% | 49.13% | 49.13% | 0.87 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | nn | NN | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 19 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 19 | -1.26 |
| BTC Market Hours Daily | xgb | XGBoost | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 19 | -1.79 |
| BTC Market Hours Daily | lstm | LSTM | 230 | 93 | 137 | 40.43% | 40.43% | 40.43% | 9.57 pp | -44 | 19 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 200 | 92 | 108 | 46.00% | 46.00% | 46.00% | 4.00 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 200 | 92 | 108 | 46.00% | 46.00% | 46.00% | 4.00 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 13 | -2.00 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
