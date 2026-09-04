# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T21:21:29.367474+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 216 | 156 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 252 | 192 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 345 | 180 | 165 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 345 | 180 | 165 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 155 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 7 | 1.14 |
| BTC Market Hours | nn | NN | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 14 | 0.43 |
| BTC Market Hours Daily | transformer | Transformer | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 15 | 0.40 |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 15 | -0.27 |
| BTC Market Hours | transformer | Transformer | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 14 | -0.29 |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 180 | 85 | 95 | 47.22% | 47.22% | 47.22% | 2.78 pp | -10 | 15 | -0.67 |
| BTC Hourly | transformer | Transformer | 156 | 75 | 81 | 48.08% | 48.08% | 48.08% | 1.92 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | rf | RandomForest | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 15 | -1.60 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 15 | -2.00 |
| BTC Market Hours | lstm | LSTM | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| BTC Daily | nn | NN | 182 | 81 | 101 | 44.51% | 44.51% | 44.51% | 5.49 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 180 | 71 | 109 | 39.44% | 39.44% | 39.44% | 10.56 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 3 | -2.67 |
| BTC Daily | transformer | Transformer | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 9 | -5.78 |
| BTC Hourly | lstm | LSTM | 156 | 57 | 99 | 36.54% | 36.54% | 36.54% | 13.46 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 156 | 56 | 100 | 35.90% | 35.90% | 35.90% | 14.10 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 182 | 63 | 119 | 34.62% | 34.62% | 34.62% | 15.38 pp | -56 | 8 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 7 | 1.14 |
| BTC Hourly | transformer | Transformer | 156 | 75 | 81 | 48.08% | 48.08% | 48.08% | 1.92 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 156 | 57 | 99 | 36.54% | 36.54% | 36.54% | 13.46 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 156 | 56 | 100 | 35.90% | 35.90% | 35.90% | 14.10 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 8 | -1.25 |
| BTC Daily | nn | NN | 182 | 81 | 101 | 44.51% | 44.51% | 44.51% | 5.49 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 9 | -5.78 |
| BTC Daily | lstm | LSTM | 182 | 63 | 119 | 34.62% | 34.62% | 34.62% | 15.38 pp | -56 | 8 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 14 | 0.43 |
| BTC Market Hours | transformer | Transformer | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 14 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | rf | RandomForest | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 14 | -1.86 |
| BTC Market Hours | lstm | LSTM | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 15 | 0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 180 | 85 | 95 | 47.22% | 47.22% | 47.22% | 2.78 pp | -10 | 15 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 15 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 180 | 71 | 109 | 39.44% | 39.44% | 39.44% | 10.56 pp | -38 | 15 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
