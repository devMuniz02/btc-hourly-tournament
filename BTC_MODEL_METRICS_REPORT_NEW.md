# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T17:37:10.441906+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 198 | 138 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 233 | 173 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 309 | 161 | 148 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 309 | 161 | 148 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| BTC Market Hours | nn | NN | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 13 | 0.38 |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 14 | -0.21 |
| BTC Hourly | transformer | Transformer | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 163 | 76 | 87 | 46.63% | 46.63% | 46.63% | 3.37 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 14 | -2.07 |
| BTC Market Hours | lstm | LSTM | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Daily | nn | NN | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | lstm | LSTM | 161 | 62 | 99 | 38.51% | 38.51% | 38.51% | 11.49 pp | -37 | 14 | -2.64 |
| BTC Hourly | nn | NN | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 6 | -2.67 |
| BTC Daily | transformer | Transformer | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 138 | 59 | 79 | 42.75% | 42.75% | 42.75% | 7.25 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 163 | 66 | 97 | 40.49% | 40.49% | 40.49% | 9.51 pp | -31 | 8 | -3.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 173 | 62 | 111 | 35.84% | 35.84% | 35.84% | 14.16 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 138 | 51 | 87 | 36.96% | 36.96% | 36.96% | 13.04 pp | -36 | 6 | -6.00 |
| BTC Daily | lstm | LSTM | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 138 | 50 | 88 | 36.23% | 36.23% | 36.23% | 13.77 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 6 | -2.67 |
| BTC Hourly | rf | RandomForest | 138 | 59 | 79 | 42.75% | 42.75% | 42.75% | 7.25 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 138 | 51 | 87 | 36.96% | 36.96% | 36.96% | 13.04 pp | -36 | 6 | -6.00 |
| BTC Hourly | lstm | LSTM | 138 | 50 | 88 | 36.23% | 36.23% | 36.23% | 13.77 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 163 | 76 | 87 | 46.63% | 46.63% | 46.63% | 3.37 pp | -11 | 8 | -1.38 |
| BTC Daily | nn | NN | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 8 | -3.12 |
| BTC Daily | rf | RandomForest | 163 | 66 | 97 | 40.49% | 40.49% | 40.49% | 9.51 pp | -31 | 8 | -3.88 |
| BTC Daily | xgb | XGBoost | 173 | 62 | 111 | 35.84% | 35.84% | 35.84% | 14.16 pp | -49 | 9 | -5.44 |
| BTC Daily | lstm | LSTM | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 8 | -6.12 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 13 | 0.38 |
| BTC Market Hours | rf | RandomForest | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 13 | -1.31 |
| BTC Market Hours | lstm | LSTM | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 14 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 14 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 161 | 62 | 99 | 38.51% | 38.51% | 38.51% | 11.49 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
