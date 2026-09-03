# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T21:28:01.742894+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 200 | 140 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 236 | 176 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 20:00:00+00:00 | 316 | 164 | 152 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 20:00:00+00:00 | 316 | 164 | 152 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 139 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 6 | 0.67 |
| BTC Market Hours | nn | NN | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 13 | 0.31 |
| Consolidated Hourly | rf | RandomForest | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 11 | 0.18 |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 6 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 138 | 66 | 72 | 47.83% | 47.83% | 47.83% | 2.17 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 138 | 66 | 72 | 47.83% | 47.83% | 47.83% | 2.17 pp | -6 | 11 | -0.55 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 13 | -0.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 13 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| BTC Market Hours | lstm | LSTM | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 13 | -2.31 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 6 | -3.00 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 166 | 67 | 99 | 40.36% | 40.36% | 40.36% | 9.64 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 166 | 58 | 108 | 34.94% | 34.94% | 34.94% | 15.06 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 140 | 50 | 90 | 35.71% | 35.71% | 35.71% | 14.29 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 6 | 0.67 |
| BTC Hourly | transformer | Transformer | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 6 | -3.00 |
| BTC Hourly | rf | RandomForest | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 140 | 50 | 90 | 35.71% | 35.71% | 35.71% | 14.29 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 166 | 73 | 93 | 43.98% | 43.98% | 43.98% | 6.02 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 166 | 67 | 99 | 40.36% | 40.36% | 40.36% | 9.64 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 166 | 58 | 108 | 34.94% | 34.94% | 34.94% | 15.06 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 13 | 0.31 |
| BTC Market Hours | rf | RandomForest | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 13 | -0.92 |
| BTC Market Hours | transformer | Transformer | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 164 | 78 | 86 | 47.56% | 47.56% | 47.56% | 2.44 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 14 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 164 | 68 | 96 | 41.46% | 41.46% | 41.46% | 8.54 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 164 | 64 | 100 | 39.02% | 39.02% | 39.02% | 10.98 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | xgb | XGBoost | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 138 | 66 | 72 | 47.83% | 47.83% | 47.83% | 2.17 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | lstm | LSTM | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 138 | 66 | 72 | 47.83% | 47.83% | 47.83% | 2.17 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
