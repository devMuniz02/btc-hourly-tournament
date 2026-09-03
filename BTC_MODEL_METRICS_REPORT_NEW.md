# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T16:15:00.325073+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 197 | 137 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 232 | 172 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 307 | 160 | 147 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 307 | 160 | 147 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 6 | 0.50 |
| BTC Market Hours | nn | NN | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 13 | 0.46 |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 137 | 68 | 69 | 49.64% | 49.64% | 49.64% | 0.36 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 162 | 78 | 84 | 48.15% | 48.15% | 48.15% | 1.85 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Market Hours | rf | RandomForest | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 162 | 75 | 87 | 46.30% | 46.30% | 46.30% | 3.70 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 14 | -1.57 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 160 | 66 | 94 | 41.25% | 41.25% | 41.25% | 8.75 pp | -28 | 14 | -2.00 |
| BTC Market Hours | lstm | LSTM | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| BTC Market Hours | xgb | XGBoost | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 6 | -2.50 |
| BTC Daily | transformer | Transformer | 162 | 71 | 91 | 43.83% | 43.83% | 43.83% | 6.17 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 160 | 62 | 98 | 38.75% | 38.75% | 38.75% | 11.25 pp | -36 | 14 | -2.57 |
| BTC Daily | rf | RandomForest | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 162 | 61 | 101 | 37.65% | 37.65% | 37.65% | 12.35 pp | -40 | 8 | -5.00 |
| BTC Daily | xgb | XGBoost | 172 | 63 | 109 | 36.63% | 36.63% | 36.63% | 13.37 pp | -46 | 9 | -5.11 |
| BTC Hourly | xgb | XGBoost | 137 | 51 | 86 | 37.23% | 37.23% | 37.23% | 12.77 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 137 | 49 | 88 | 35.77% | 35.77% | 35.77% | 14.23 pp | -39 | 6 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 6 | 0.50 |
| BTC Hourly | transformer | Transformer | 137 | 68 | 69 | 49.64% | 49.64% | 49.64% | 0.36 pp | -1 | 6 | -0.17 |
| BTC Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 6 | -2.50 |
| BTC Hourly | rf | RandomForest | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 137 | 51 | 86 | 37.23% | 37.23% | 37.23% | 12.77 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 137 | 49 | 88 | 35.77% | 35.77% | 35.77% | 14.23 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 162 | 78 | 84 | 48.15% | 48.15% | 48.15% | 1.85 pp | -6 | 8 | -0.75 |
| BTC Daily | nn | NN | 162 | 75 | 87 | 46.30% | 46.30% | 46.30% | 3.70 pp | -12 | 8 | -1.50 |
| BTC Daily | transformer | Transformer | 162 | 71 | 91 | 43.83% | 43.83% | 43.83% | 6.17 pp | -20 | 8 | -2.50 |
| BTC Daily | rf | RandomForest | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 8 | -3.25 |
| BTC Daily | lstm | LSTM | 162 | 61 | 101 | 37.65% | 37.65% | 37.65% | 12.35 pp | -40 | 8 | -5.00 |
| BTC Daily | xgb | XGBoost | 172 | 63 | 109 | 36.63% | 36.63% | 36.63% | 13.37 pp | -46 | 9 | -5.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 13 | 0.46 |
| BTC Market Hours | rf | RandomForest | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 13 | -1.23 |
| BTC Market Hours | lstm | LSTM | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| BTC Market Hours | xgb | XGBoost | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 14 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 160 | 66 | 94 | 41.25% | 41.25% | 41.25% | 8.75 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 160 | 62 | 98 | 38.75% | 38.75% | 38.75% | 11.25 pp | -36 | 14 | -2.57 |

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
