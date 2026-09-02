# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T18:32:52.993053+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 182 | 122 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 218 | 158 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 282 | 146 | 136 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 281 | 145 | 136 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| BTC Market Hours | nn | NN | 146 | 76 | 70 | 52.05% | 52.05% | 52.05% | 2.05 pp | 6 | 12 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 148 | 73 | 75 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 145 | 70 | 75 | 48.28% | 48.28% | 48.28% | 1.72 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 13 | -0.54 |
| BTC Hourly | transformer | Transformer | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 6 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| BTC Market Hours | rf | RandomForest | 146 | 68 | 78 | 46.58% | 46.58% | 46.58% | 3.42 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 145 | 64 | 81 | 44.14% | 44.14% | 44.14% | 5.86 pp | -17 | 13 | -1.31 |
| BTC Hourly | nn | NN | 122 | 57 | 65 | 46.72% | 46.72% | 46.72% | 3.28 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 12 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Daily | nn | NN | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 7 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 146 | 61 | 85 | 41.78% | 41.78% | 41.78% | 8.22 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 145 | 59 | 86 | 40.69% | 40.69% | 40.69% | 9.31 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 148 | 66 | 82 | 44.59% | 44.59% | 44.59% | 5.41 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Market Hours | lstm | LSTM | 146 | 58 | 88 | 39.73% | 39.73% | 39.73% | 10.27 pp | -30 | 12 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 145 | 55 | 90 | 37.93% | 37.93% | 37.93% | 12.07 pp | -35 | 13 | -2.69 |
| BTC Hourly | rf | RandomForest | 122 | 51 | 71 | 41.80% | 41.80% | 41.80% | 8.20 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 148 | 62 | 86 | 41.89% | 41.89% | 41.89% | 8.11 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 122 | 46 | 76 | 37.70% | 37.70% | 37.70% | 12.30 pp | -30 | 6 | -5.00 |
| BTC Daily | xgb | XGBoost | 158 | 58 | 100 | 36.71% | 36.71% | 36.71% | 13.29 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 148 | 52 | 96 | 35.14% | 35.14% | 35.14% | 14.86 pp | -44 | 7 | -6.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| BTC Hourly | lstm | LSTM | 122 | 41 | 81 | 33.61% | 33.61% | 33.61% | 16.39 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 122 | 57 | 65 | 46.72% | 46.72% | 46.72% | 3.28 pp | -8 | 6 | -1.33 |
| BTC Hourly | rf | RandomForest | 122 | 51 | 71 | 41.80% | 41.80% | 41.80% | 8.20 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 122 | 46 | 76 | 37.70% | 37.70% | 37.70% | 12.30 pp | -30 | 6 | -5.00 |
| BTC Hourly | lstm | LSTM | 122 | 41 | 81 | 33.61% | 33.61% | 33.61% | 16.39 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 148 | 73 | 75 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| BTC Daily | nn | NN | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 7 | -2.00 |
| BTC Daily | transformer | Transformer | 148 | 66 | 82 | 44.59% | 44.59% | 44.59% | 5.41 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 148 | 62 | 86 | 41.89% | 41.89% | 41.89% | 8.11 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 158 | 58 | 100 | 36.71% | 36.71% | 36.71% | 13.29 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 148 | 52 | 96 | 35.14% | 35.14% | 35.14% | 14.86 pp | -44 | 7 | -6.29 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 146 | 76 | 70 | 52.05% | 52.05% | 52.05% | 2.05 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 146 | 68 | 78 | 46.58% | 46.58% | 46.58% | 3.42 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 12 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 146 | 61 | 85 | 41.78% | 41.78% | 41.78% | 8.22 pp | -24 | 12 | -2.00 |
| BTC Market Hours | lstm | LSTM | 146 | 58 | 88 | 39.73% | 39.73% | 39.73% | 10.27 pp | -30 | 12 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 145 | 70 | 75 | 48.28% | 48.28% | 48.28% | 1.72 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | transformer | Transformer | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 145 | 64 | 81 | 44.14% | 44.14% | 44.14% | 5.86 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 145 | 59 | 86 | 40.69% | 40.69% | 40.69% | 9.31 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 145 | 55 | 90 | 37.93% | 37.93% | 37.93% | 12.07 pp | -35 | 13 | -2.69 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
