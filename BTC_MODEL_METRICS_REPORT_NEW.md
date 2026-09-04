# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T23:13:24.403115+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 217 | 157 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 253 | 193 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 22:00:00+00:00 | 348 | 181 | 167 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 22:00:00+00:00 | 348 | 181 | 167 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 155 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 157 | 82 | 75 | 52.23% | 52.23% | 52.23% | 2.23 pp | 7 | 7 | 1.00 |
| BTC Market Hours Daily | transformer | Transformer | 181 | 94 | 87 | 51.93% | 51.93% | 51.93% | 1.93 pp | 7 | 15 | 0.47 |
| BTC Market Hours | nn | NN | 181 | 93 | 88 | 51.38% | 51.38% | 51.38% | 1.38 pp | 5 | 14 | 0.36 |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| BTC Market Hours | transformer | Transformer | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 14 | -0.21 |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 157 | 76 | 81 | 48.41% | 48.41% | 48.41% | 1.59 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 181 | 85 | 96 | 46.96% | 46.96% | 46.96% | 3.04 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 15 | -1.53 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 181 | 75 | 106 | 41.44% | 41.44% | 41.44% | 8.56 pp | -31 | 15 | -2.07 |
| BTC Market Hours | lstm | LSTM | 181 | 76 | 105 | 41.99% | 41.99% | 41.99% | 8.01 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| BTC Daily | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 181 | 72 | 109 | 39.78% | 39.78% | 39.78% | 10.22 pp | -37 | 15 | -2.47 |
| BTC Daily | transformer | Transformer | 183 | 80 | 103 | 43.72% | 43.72% | 43.72% | 6.28 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 7 | -3.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 183 | 73 | 110 | 39.89% | 39.89% | 39.89% | 10.11 pp | -37 | 8 | -4.62 |
| BTC Daily | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 9 | -5.89 |
| BTC Hourly | lstm | LSTM | 157 | 57 | 100 | 36.31% | 36.31% | 36.31% | 13.69 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 157 | 56 | 101 | 35.67% | 35.67% | 35.67% | 14.33 pp | -45 | 7 | -6.43 |
| BTC Daily | lstm | LSTM | 183 | 63 | 120 | 34.43% | 34.43% | 34.43% | 15.57 pp | -57 | 8 | -7.12 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 157 | 82 | 75 | 52.23% | 52.23% | 52.23% | 2.23 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 157 | 76 | 81 | 48.41% | 48.41% | 48.41% | 1.59 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 7 | -3.86 |
| BTC Hourly | lstm | LSTM | 157 | 57 | 100 | 36.31% | 36.31% | 36.31% | 13.69 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 157 | 56 | 101 | 35.67% | 35.67% | 35.67% | 14.33 pp | -45 | 7 | -6.43 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 8 | -1.38 |
| BTC Daily | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 8 | -2.38 |
| BTC Daily | transformer | Transformer | 183 | 80 | 103 | 43.72% | 43.72% | 43.72% | 6.28 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 183 | 73 | 110 | 39.89% | 39.89% | 39.89% | 10.11 pp | -37 | 8 | -4.62 |
| BTC Daily | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 9 | -5.89 |
| BTC Daily | lstm | LSTM | 183 | 63 | 120 | 34.43% | 34.43% | 34.43% | 15.57 pp | -57 | 8 | -7.12 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 181 | 93 | 88 | 51.38% | 51.38% | 51.38% | 1.38 pp | 5 | 14 | 0.36 |
| BTC Market Hours | transformer | Transformer | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 14 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 14 | -1.07 |
| BTC Market Hours | xgb | XGBoost | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 181 | 76 | 105 | 41.99% | 41.99% | 41.99% | 8.01 pp | -29 | 14 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 181 | 94 | 87 | 51.93% | 51.93% | 51.93% | 1.93 pp | 7 | 15 | 0.47 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 181 | 85 | 96 | 46.96% | 46.96% | 46.96% | 3.04 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 15 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 181 | 75 | 106 | 41.44% | 41.44% | 41.44% | 8.56 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 181 | 72 | 109 | 39.78% | 39.78% | 39.78% | 10.22 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
