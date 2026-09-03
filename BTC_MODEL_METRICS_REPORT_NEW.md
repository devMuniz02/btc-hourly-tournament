# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T18:33:19.239037+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 138 | 70 | 68 | 50.72% | 50.72% | 50.72% | 0.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours | nn | NN | 162 | 83 | 79 | 51.23% | 51.23% | 51.23% | 1.23 pp | 4 | 13 | 0.31 |
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 162 | 77 | 85 | 47.53% | 47.53% | 47.53% | 2.47 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours | rf | RandomForest | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 13 | -1.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 164 | 77 | 87 | 46.95% | 46.95% | 46.95% | 3.05 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 162 | 70 | 92 | 43.21% | 43.21% | 43.21% | 6.79 pp | -22 | 14 | -1.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 14 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 162 | 67 | 95 | 41.36% | 41.36% | 41.36% | 8.64 pp | -28 | 13 | -2.15 |
| BTC Market Hours | lstm | LSTM | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 13 | -2.31 |
| Consolidated Market Hours Daily | nn | NN | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| BTC Daily | nn | NN | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 162 | 63 | 99 | 38.89% | 38.89% | 38.89% | 11.11 pp | -36 | 14 | -2.57 |
| BTC Hourly | nn | NN | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Daily | transformer | Transformer | 164 | 70 | 94 | 42.68% | 42.68% | 42.68% | 7.32 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 138 | 59 | 79 | 42.75% | 42.75% | 42.75% | 7.25 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 164 | 67 | 97 | 40.85% | 40.85% | 40.85% | 9.15 pp | -30 | 8 | -3.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
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
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
