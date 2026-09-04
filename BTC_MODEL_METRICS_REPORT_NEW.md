# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T19:06:55.405785+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 214 | 154 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 250 | 190 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 341 | 178 | 163 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 341 | 178 | 163 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 154 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 154 | 81 | 73 | 52.60% | 52.60% | 52.60% | 2.60 pp | 8 | 7 | 1.14 |
| BTC Market Hours | nn | NN | 178 | 93 | 85 | 52.25% | 52.25% | 52.25% | 2.25 pp | 8 | 14 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 15 | 0.27 |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 15 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 14 | -0.43 |
| BTC Market Hours Daily | nn | NN | 178 | 84 | 94 | 47.19% | 47.19% | 47.19% | 2.81 pp | -10 | 15 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 154 | 74 | 80 | 48.05% | 48.05% | 48.05% | 1.95 pp | -6 | 7 | -0.86 |
| BTC Market Hours | rf | RandomForest | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 178 | 80 | 98 | 44.94% | 44.94% | 44.94% | 5.06 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 15 | -1.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 180 | 84 | 96 | 46.67% | 46.67% | 46.67% | 3.33 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| BTC Market Hours Daily | xgb | XGBoost | 178 | 74 | 104 | 41.57% | 41.57% | 41.57% | 8.43 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 178 | 73 | 105 | 41.01% | 41.01% | 41.01% | 8.99 pp | -32 | 14 | -2.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 3 | -2.33 |
| BTC Daily | nn | NN | 180 | 80 | 100 | 44.44% | 44.44% | 44.44% | 5.56 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 178 | 70 | 108 | 39.33% | 39.33% | 39.33% | 10.67 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Daily | transformer | Transformer | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 180 | 73 | 107 | 40.56% | 40.56% | 40.56% | 9.44 pp | -34 | 8 | -4.25 |
| BTC Daily | xgb | XGBoost | 190 | 69 | 121 | 36.32% | 36.32% | 36.32% | 13.68 pp | -52 | 9 | -5.78 |
| BTC Hourly | lstm | LSTM | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 154 | 55 | 99 | 35.71% | 35.71% | 35.71% | 14.29 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 180 | 63 | 117 | 35.00% | 35.00% | 35.00% | 15.00 pp | -54 | 8 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 154 | 81 | 73 | 52.60% | 52.60% | 52.60% | 2.60 pp | 8 | 7 | 1.14 |
| BTC Hourly | transformer | Transformer | 154 | 74 | 80 | 48.05% | 48.05% | 48.05% | 1.95 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 154 | 55 | 99 | 35.71% | 35.71% | 35.71% | 14.29 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 180 | 84 | 96 | 46.67% | 46.67% | 46.67% | 3.33 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 180 | 80 | 100 | 44.44% | 44.44% | 44.44% | 5.56 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 180 | 73 | 107 | 40.56% | 40.56% | 40.56% | 9.44 pp | -34 | 8 | -4.25 |
| BTC Daily | xgb | XGBoost | 190 | 69 | 121 | 36.32% | 36.32% | 36.32% | 13.68 pp | -52 | 9 | -5.78 |
| BTC Daily | lstm | LSTM | 180 | 63 | 117 | 35.00% | 35.00% | 35.00% | 15.00 pp | -54 | 8 | -6.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 178 | 93 | 85 | 52.25% | 52.25% | 52.25% | 2.25 pp | 8 | 14 | 0.57 |
| BTC Market Hours | transformer | Transformer | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 14 | -0.43 |
| BTC Market Hours | rf | RandomForest | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 178 | 80 | 98 | 44.94% | 44.94% | 44.94% | 5.06 pp | -18 | 14 | -1.29 |
| BTC Market Hours | xgb | XGBoost | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 178 | 73 | 105 | 41.01% | 41.01% | 41.01% | 8.99 pp | -32 | 14 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 15 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 178 | 84 | 94 | 47.19% | 47.19% | 47.19% | 2.81 pp | -10 | 15 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 15 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 178 | 74 | 104 | 41.57% | 41.57% | 41.57% | 8.43 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 178 | 70 | 108 | 39.33% | 39.33% | 39.33% | 10.67 pp | -38 | 15 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
