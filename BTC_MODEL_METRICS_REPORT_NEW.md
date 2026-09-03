# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T18:07:20.204614+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 234 | 174 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 17:00:00+00:00 | 311 | 162 | 149 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 17:00:00+00:00 | 311 | 162 | 149 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 136 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 136 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 136 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 137 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours | nn | NN | 162 | 83 | 79 | 51.23% | 51.23% | 51.23% | 1.23 pp | 4 | 13 | 0.31 |
| Consolidated Hourly | rf | RandomForest | 136 | 68 | 68 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 136 | 68 | 68 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 162 | 77 | 85 | 47.53% | 47.53% | 47.53% | 2.47 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Market Hours | rf | RandomForest | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 11 | -1.09 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | nn | NN | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 13 | -1.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 164 | 77 | 87 | 46.95% | 46.95% | 46.95% | 3.05 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 162 | 70 | 92 | 43.21% | 43.21% | 43.21% | 6.79 pp | -22 | 14 | -1.57 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 14 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 162 | 67 | 95 | 41.36% | 41.36% | 41.36% | 8.64 pp | -28 | 13 | -2.15 |
| BTC Market Hours | lstm | LSTM | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 13 | -2.31 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Daily | nn | NN | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 162 | 63 | 99 | 38.89% | 38.89% | 38.89% | 11.11 pp | -36 | 14 | -2.57 |
| BTC Hourly | nn | NN | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 6 | -2.67 |
| BTC Daily | transformer | Transformer | 164 | 70 | 94 | 42.68% | 42.68% | 42.68% | 7.32 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 138 | 59 | 79 | 42.75% | 42.75% | 42.75% | 7.25 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 8 | -3.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 9 | -5.33 |
| BTC Hourly | xgb | XGBoost | 138 | 51 | 87 | 36.96% | 36.96% | 36.96% | 13.04 pp | -36 | 6 | -6.00 |
| BTC Daily | lstm | LSTM | 164 | 57 | 107 | 34.76% | 34.76% | 34.76% | 15.24 pp | -50 | 8 | -6.25 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 164 | 77 | 87 | 46.95% | 46.95% | 46.95% | 3.05 pp | -10 | 8 | -1.25 |
| BTC Daily | nn | NN | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 164 | 70 | 94 | 42.68% | 42.68% | 42.68% | 7.32 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 8 | -3.75 |
| BTC Daily | xgb | XGBoost | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 164 | 57 | 107 | 34.76% | 34.76% | 34.76% | 15.24 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 162 | 83 | 79 | 51.23% | 51.23% | 51.23% | 1.23 pp | 4 | 13 | 0.31 |
| BTC Market Hours | rf | RandomForest | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 162 | 67 | 95 | 41.36% | 41.36% | 41.36% | 8.64 pp | -28 | 13 | -2.15 |
| BTC Market Hours | lstm | LSTM | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 162 | 77 | 85 | 47.53% | 47.53% | 47.53% | 2.47 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 162 | 70 | 92 | 43.21% | 43.21% | 43.21% | 6.79 pp | -22 | 14 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 14 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 162 | 63 | 99 | 38.89% | 38.89% | 38.89% | 11.11 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 136 | 68 | 68 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | lstm | LSTM | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 136 | 68 | 68 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 11 | -1.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
