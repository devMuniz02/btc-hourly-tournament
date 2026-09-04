# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T06:46:38.123308+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 206 | 146 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 242 | 182 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 326 | 170 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 326 | 170 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 146 | 76 | 70 | 52.05% | 52.05% | 52.05% | 2.05 pp | 6 | 7 | 0.86 |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 170 | 88 | 82 | 51.76% | 51.76% | 51.76% | 1.76 pp | 6 | 14 | 0.43 |
| Consolidated Hourly | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| BTC Market Hours Daily | transformer | Transformer | 170 | 84 | 86 | 49.41% | 49.41% | 49.41% | 0.59 pp | -2 | 14 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| BTC Hourly | transformer | Transformer | 146 | 72 | 74 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 14 | -0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours Daily | nn | NN | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 14 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 14 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 170 | 71 | 99 | 41.76% | 41.76% | 41.76% | 8.24 pp | -28 | 14 | -2.00 |
| BTC Market Hours | lstm | LSTM | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 14 | -2.14 |
| BTC Daily | nn | NN | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 8 | -2.25 |
| Consolidated Hourly | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 170 | 67 | 103 | 39.41% | 39.41% | 39.41% | 10.59 pp | -36 | 14 | -2.57 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Daily | transformer | Transformer | 172 | 75 | 97 | 43.60% | 43.60% | 43.60% | 6.40 pp | -22 | 8 | -2.75 |
| BTC Hourly | nn | NN | 146 | 63 | 83 | 43.15% | 43.15% | 43.15% | 6.85 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 8 | -3.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 182 | 68 | 114 | 37.36% | 37.36% | 37.36% | 12.64 pp | -46 | 9 | -5.11 |
| BTC Hourly | lstm | LSTM | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |
| BTC Daily | lstm | LSTM | 172 | 60 | 112 | 34.88% | 34.88% | 34.88% | 15.12 pp | -52 | 8 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 146 | 76 | 70 | 52.05% | 52.05% | 52.05% | 2.05 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 146 | 72 | 74 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| BTC Hourly | nn | NN | 146 | 63 | 83 | 43.15% | 43.15% | 43.15% | 6.85 pp | -20 | 7 | -2.86 |
| BTC Hourly | rf | RandomForest | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 172 | 75 | 97 | 43.60% | 43.60% | 43.60% | 6.40 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 8 | -3.75 |
| BTC Daily | xgb | XGBoost | 182 | 68 | 114 | 37.36% | 37.36% | 37.36% | 12.64 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 172 | 60 | 112 | 34.88% | 34.88% | 34.88% | 15.12 pp | -52 | 8 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 170 | 88 | 82 | 51.76% | 51.76% | 51.76% | 1.76 pp | 6 | 14 | 0.43 |
| BTC Market Hours | rf | RandomForest | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 14 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 170 | 84 | 86 | 49.41% | 49.41% | 49.41% | 0.59 pp | -2 | 14 | -0.14 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 14 | -0.43 |
| BTC Market Hours Daily | nn | NN | 170 | 79 | 91 | 46.47% | 46.47% | 46.47% | 3.53 pp | -12 | 14 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 14 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 170 | 71 | 99 | 41.76% | 41.76% | 41.76% | 8.24 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 170 | 67 | 103 | 39.41% | 39.41% | 39.41% | 10.59 pp | -36 | 14 | -2.57 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
