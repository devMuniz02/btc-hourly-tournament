# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T18:11:45.136695+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 340 | 178 | 162 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 339 | 177 | 162 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 154 | 81 | 73 | 52.60% | 52.60% | 52.60% | 2.60 pp | 8 | 7 | 1.14 |
| BTC Market Hours | nn | NN | 178 | 93 | 85 | 52.25% | 52.25% | 52.25% | 2.25 pp | 8 | 14 | 0.57 |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| BTC Market Hours Daily | transformer | Transformer | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 15 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 15 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| BTC Market Hours | transformer | Transformer | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 14 | -0.43 |
| BTC Market Hours Daily | nn | NN | 177 | 83 | 94 | 46.89% | 46.89% | 46.89% | 3.11 pp | -11 | 15 | -0.73 |
| BTC Hourly | transformer | Transformer | 154 | 74 | 80 | 48.05% | 48.05% | 48.05% | 1.95 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours | rf | RandomForest | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 178 | 80 | 98 | 44.94% | 44.94% | 44.94% | 5.06 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 15 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 180 | 84 | 96 | 46.67% | 46.67% | 46.67% | 3.33 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 15 | -2.07 |
| BTC Market Hours | lstm | LSTM | 178 | 73 | 105 | 41.01% | 41.01% | 41.01% | 8.99 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 177 | 70 | 107 | 39.55% | 39.55% | 39.55% | 10.45 pp | -37 | 15 | -2.47 |
| BTC Daily | nn | NN | 180 | 80 | 100 | 44.44% | 44.44% | 44.44% | 5.56 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
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
| BTC Market Hours Daily | transformer | Transformer | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 15 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 15 | -0.20 |
| BTC Market Hours Daily | nn | NN | 177 | 83 | 94 | 46.89% | 46.89% | 46.89% | 3.11 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 15 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 177 | 70 | 107 | 39.55% | 39.55% | 39.55% | 10.45 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
