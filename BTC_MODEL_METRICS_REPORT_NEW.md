# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T19:16:05.855530+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 199 | 139 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 234 | 174 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 18:00:00+00:00 | 312 | 162 | 150 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 18:00:00+00:00 | 312 | 162 | 150 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 137 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 137 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 30 | 107 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 30 | 107 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| BTC Market Hours | nn | NN | 162 | 83 | 79 | 51.23% | 51.23% | 51.23% | 1.23 pp | 4 | 13 | 0.31 |
| BTC Market Hours Daily | transformer | Transformer | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Hourly | transformer | Transformer | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 6 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 162 | 77 | 85 | 47.53% | 47.53% | 47.53% | 2.47 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| BTC Market Hours | rf | RandomForest | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 162 | 73 | 89 | 45.06% | 45.06% | 45.06% | 4.94 pp | -16 | 13 | -1.23 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 162 | 70 | 92 | 43.21% | 43.21% | 43.21% | 6.79 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 14 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 162 | 67 | 95 | 41.36% | 41.36% | 41.36% | 8.64 pp | -28 | 13 | -2.15 |
| BTC Market Hours | lstm | LSTM | 162 | 66 | 96 | 40.74% | 40.74% | 40.74% | 9.26 pp | -30 | 13 | -2.31 |
| BTC Daily | nn | NN | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 162 | 63 | 99 | 38.89% | 38.89% | 38.89% | 11.11 pp | -36 | 14 | -2.57 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 6 | -2.83 |
| BTC Daily | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 6 | -3.50 |
| BTC Daily | rf | RandomForest | 164 | 66 | 98 | 40.24% | 40.24% | 40.24% | 9.76 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 174 | 62 | 112 | 35.63% | 35.63% | 35.63% | 14.37 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 6 | -6.17 |
| BTC Daily | lstm | LSTM | 164 | 57 | 107 | 34.76% | 34.76% | 34.76% | 15.24 pp | -50 | 8 | -6.25 |
| BTC Hourly | lstm | LSTM | 139 | 50 | 89 | 35.97% | 35.97% | 35.97% | 14.03 pp | -39 | 6 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 6 | 0.50 |
| BTC Hourly | transformer | Transformer | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 6 | -2.83 |
| BTC Hourly | rf | RandomForest | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 6 | -6.17 |
| BTC Hourly | lstm | LSTM | 139 | 50 | 89 | 35.97% | 35.97% | 35.97% | 14.03 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 164 | 76 | 88 | 46.34% | 46.34% | 46.34% | 3.66 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 164 | 72 | 92 | 43.90% | 43.90% | 43.90% | 6.10 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 8 | -3.25 |
| BTC Daily | rf | RandomForest | 164 | 66 | 98 | 40.24% | 40.24% | 40.24% | 9.76 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 174 | 62 | 112 | 35.63% | 35.63% | 35.63% | 14.37 pp | -50 | 9 | -5.56 |
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
| Consolidated Hourly | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
