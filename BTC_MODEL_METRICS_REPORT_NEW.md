# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T18:10:37.797092+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 181 | 121 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 217 | 157 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 281 | 145 | 136 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 281 | 145 | 136 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 12 | 0.58 |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 6 | 0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 147 | 73 | 74 | 49.66% | 49.66% | 49.66% | 0.34 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 145 | 70 | 75 | 48.28% | 48.28% | 48.28% | 1.72 pp | -5 | 13 | -0.38 |
| BTC Hourly | transformer | Transformer | 121 | 59 | 62 | 48.76% | 48.76% | 48.76% | 1.24 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| BTC Market Hours | rf | RandomForest | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 12 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 13 | -1.15 |
| BTC Hourly | nn | NN | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 6 | -1.17 |
| BTC Market Hours | transformer | Transformer | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 145 | 64 | 81 | 44.14% | 44.14% | 44.14% | 5.86 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 145 | 59 | 86 | 40.69% | 40.69% | 40.69% | 9.31 pp | -27 | 13 | -2.08 |
| BTC Daily | nn | NN | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Daily | transformer | Transformer | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 145 | 57 | 88 | 39.31% | 39.31% | 39.31% | 10.69 pp | -31 | 12 | -2.58 |
| BTC Market Hours Daily | lstm | LSTM | 145 | 55 | 90 | 37.93% | 37.93% | 37.93% | 12.07 pp | -35 | 13 | -2.69 |
| BTC Hourly | rf | RandomForest | 121 | 51 | 70 | 42.15% | 42.15% | 42.15% | 7.85 pp | -19 | 6 | -3.17 |
| BTC Daily | rf | RandomForest | 147 | 62 | 85 | 42.18% | 42.18% | 42.18% | 7.82 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 121 | 46 | 75 | 38.02% | 38.02% | 38.02% | 11.98 pp | -29 | 6 | -4.83 |
| BTC Daily | xgb | XGBoost | 157 | 58 | 99 | 36.94% | 36.94% | 36.94% | 13.06 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 147 | 52 | 95 | 35.37% | 35.37% | 35.37% | 14.63 pp | -43 | 7 | -6.14 |
| BTC Hourly | lstm | LSTM | 121 | 41 | 80 | 33.88% | 33.88% | 33.88% | 16.12 pp | -39 | 6 | -6.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 6 | 0.17 |
| BTC Hourly | transformer | Transformer | 121 | 59 | 62 | 48.76% | 48.76% | 48.76% | 1.24 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 6 | -1.17 |
| BTC Hourly | rf | RandomForest | 121 | 51 | 70 | 42.15% | 42.15% | 42.15% | 7.85 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 121 | 46 | 75 | 38.02% | 38.02% | 38.02% | 11.98 pp | -29 | 6 | -4.83 |
| BTC Hourly | lstm | LSTM | 121 | 41 | 80 | 33.88% | 33.88% | 33.88% | 16.12 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 147 | 73 | 74 | 49.66% | 49.66% | 49.66% | 0.34 pp | -1 | 7 | -0.14 |
| BTC Daily | nn | NN | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 147 | 62 | 85 | 42.18% | 42.18% | 42.18% | 7.82 pp | -23 | 7 | -3.29 |
| BTC Daily | xgb | XGBoost | 157 | 58 | 99 | 36.94% | 36.94% | 36.94% | 13.06 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 147 | 52 | 95 | 35.37% | 35.37% | 35.37% | 14.63 pp | -43 | 7 | -6.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 12 | 0.58 |
| BTC Market Hours | rf | RandomForest | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 12 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 12 | -1.08 |
| BTC Market Hours | transformer | Transformer | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 12 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 145 | 57 | 88 | 39.31% | 39.31% | 39.31% | 10.69 pp | -31 | 12 | -2.58 |

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
