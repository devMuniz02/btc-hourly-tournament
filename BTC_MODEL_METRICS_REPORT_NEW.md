# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T09:24:46.252383+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 176 | 116 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 212 | 152 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 270 | 140 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 270 | 140 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 117 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 140 | 67 | 73 | 47.86% | 47.86% | 47.86% | 2.14 pp | -6 | 12 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 140 | 65 | 75 | 46.43% | 46.43% | 46.43% | 3.57 pp | -10 | 12 | -0.83 |
| BTC Daily | mlp_sklearn | MLPClassifier | 142 | 68 | 74 | 47.89% | 47.89% | 47.89% | 2.11 pp | -6 | 7 | -0.86 |
| BTC Market Hours | rf | RandomForest | 140 | 65 | 75 | 46.43% | 46.43% | 46.43% | 3.57 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| BTC Hourly | transformer | Transformer | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | nn | NN | 140 | 62 | 78 | 44.29% | 44.29% | 44.29% | 5.71 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 140 | 62 | 78 | 44.29% | 44.29% | 44.29% | 5.71 pp | -16 | 12 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 116 | 54 | 62 | 46.55% | 46.55% | 46.55% | 3.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 140 | 57 | 83 | 40.71% | 40.71% | 40.71% | 9.29 pp | -26 | 12 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 140 | 58 | 82 | 41.43% | 41.43% | 41.43% | 8.57 pp | -24 | 11 | -2.18 |
| BTC Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Daily | transformer | Transformer | 142 | 62 | 80 | 43.66% | 43.66% | 43.66% | 6.34 pp | -18 | 7 | -2.57 |
| BTC Market Hours | lstm | LSTM | 140 | 55 | 85 | 39.29% | 39.29% | 39.29% | 10.71 pp | -30 | 11 | -2.73 |
| BTC Market Hours Daily | lstm | LSTM | 140 | 53 | 87 | 37.86% | 37.86% | 37.86% | 12.14 pp | -34 | 12 | -2.83 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| BTC Hourly | rf | RandomForest | 116 | 49 | 67 | 42.24% | 42.24% | 42.24% | 7.76 pp | -18 | 5 | -3.60 |
| BTC Daily | rf | RandomForest | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |
| BTC Daily | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 8 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| BTC Daily | lstm | LSTM | 142 | 49 | 93 | 34.51% | 34.51% | 34.51% | 15.49 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 116 | 42 | 74 | 36.21% | 36.21% | 36.21% | 13.79 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 116 | 37 | 79 | 31.90% | 31.90% | 31.90% | 18.10 pp | -42 | 5 | -8.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 5 | -0.40 |
| BTC Hourly | transformer | Transformer | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 5 | -1.20 |
| BTC Hourly | nn | NN | 116 | 54 | 62 | 46.55% | 46.55% | 46.55% | 3.45 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 116 | 49 | 67 | 42.24% | 42.24% | 42.24% | 7.76 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 116 | 42 | 74 | 36.21% | 36.21% | 36.21% | 13.79 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 116 | 37 | 79 | 31.90% | 31.90% | 31.90% | 18.10 pp | -42 | 5 | -8.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 142 | 68 | 74 | 47.89% | 47.89% | 47.89% | 2.11 pp | -6 | 7 | -0.86 |
| BTC Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 142 | 62 | 80 | 43.66% | 43.66% | 43.66% | 6.34 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 8 | -5.50 |
| BTC Daily | lstm | LSTM | 142 | 49 | 93 | 34.51% | 34.51% | 34.51% | 15.49 pp | -44 | 7 | -6.29 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| BTC Market Hours | rf | RandomForest | 140 | 65 | 75 | 46.43% | 46.43% | 46.43% | 3.57 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| BTC Market Hours | xgb | XGBoost | 140 | 58 | 82 | 41.43% | 41.43% | 41.43% | 8.57 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 140 | 55 | 85 | 39.29% | 39.29% | 39.29% | 10.71 pp | -30 | 11 | -2.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 140 | 67 | 73 | 47.86% | 47.86% | 47.86% | 2.14 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 140 | 65 | 75 | 46.43% | 46.43% | 46.43% | 3.57 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | nn | NN | 140 | 62 | 78 | 44.29% | 44.29% | 44.29% | 5.71 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 140 | 62 | 78 | 44.29% | 44.29% | 44.29% | 5.71 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 140 | 57 | 83 | 40.71% | 40.71% | 40.71% | 9.29 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 140 | 53 | 87 | 37.86% | 37.86% | 37.86% | 12.14 pp | -34 | 12 | -2.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
