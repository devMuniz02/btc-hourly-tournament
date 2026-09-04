# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T08:26:39.942642+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 207 | 147 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 243 | 183 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 327 | 171 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 327 | 171 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 147 | 77 | 70 | 52.38% | 52.38% | 52.38% | 2.38 pp | 7 | 7 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 171 | 89 | 82 | 52.05% | 52.05% | 52.05% | 2.05 pp | 7 | 14 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| BTC Market Hours Daily | transformer | Transformer | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 15 | -0.33 |
| BTC Hourly | transformer | Transformer | 147 | 72 | 75 | 48.98% | 48.98% | 48.98% | 1.02 pp | -3 | 7 | -0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 15 | -0.87 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| BTC Market Hours | rf | RandomForest | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| BTC Market Hours | transformer | Transformer | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 14 | -1.79 |
| BTC Daily | mlp_sklearn | MLPClassifier | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 15 | -1.93 |
| BTC Market Hours | lstm | LSTM | 171 | 70 | 101 | 40.94% | 40.94% | 40.94% | 9.06 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| BTC Daily | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 171 | 67 | 104 | 39.18% | 39.18% | 39.18% | 10.82 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 147 | 64 | 83 | 43.54% | 43.54% | 43.54% | 6.46 pp | -19 | 7 | -2.71 |
| BTC Daily | transformer | Transformer | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 173 | 71 | 102 | 41.04% | 41.04% | 41.04% | 8.96 pp | -31 | 8 | -3.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 183 | 68 | 115 | 37.16% | 37.16% | 37.16% | 12.84 pp | -47 | 9 | -5.22 |
| BTC Hourly | lstm | LSTM | 147 | 54 | 93 | 36.73% | 36.73% | 36.73% | 13.27 pp | -39 | 7 | -5.57 |
| BTC Hourly | xgb | XGBoost | 147 | 53 | 94 | 36.05% | 36.05% | 36.05% | 13.95 pp | -41 | 7 | -5.86 |
| BTC Daily | lstm | LSTM | 173 | 60 | 113 | 34.68% | 34.68% | 34.68% | 15.32 pp | -53 | 8 | -6.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 147 | 77 | 70 | 52.38% | 52.38% | 52.38% | 2.38 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 147 | 72 | 75 | 48.98% | 48.98% | 48.98% | 1.02 pp | -3 | 7 | -0.43 |
| BTC Hourly | nn | NN | 147 | 64 | 83 | 43.54% | 43.54% | 43.54% | 6.46 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 147 | 54 | 93 | 36.73% | 36.73% | 36.73% | 13.27 pp | -39 | 7 | -5.57 |
| BTC Hourly | xgb | XGBoost | 147 | 53 | 94 | 36.05% | 36.05% | 36.05% | 13.95 pp | -41 | 7 | -5.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 8 | -1.88 |
| BTC Daily | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 8 | -2.38 |
| BTC Daily | transformer | Transformer | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 173 | 71 | 102 | 41.04% | 41.04% | 41.04% | 8.96 pp | -31 | 8 | -3.88 |
| BTC Daily | xgb | XGBoost | 183 | 68 | 115 | 37.16% | 37.16% | 37.16% | 12.84 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 173 | 60 | 113 | 34.68% | 34.68% | 34.68% | 15.32 pp | -53 | 8 | -6.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 171 | 89 | 82 | 52.05% | 52.05% | 52.05% | 2.05 pp | 7 | 14 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| BTC Market Hours | rf | RandomForest | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| BTC Market Hours | transformer | Transformer | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 14 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 171 | 70 | 101 | 40.94% | 40.94% | 40.94% | 9.06 pp | -31 | 14 | -2.21 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 15 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 171 | 67 | 104 | 39.18% | 39.18% | 39.18% | 10.82 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
