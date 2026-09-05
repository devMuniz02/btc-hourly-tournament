# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T08:34:58.499308+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 224 | 164 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 260 | 200 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 357 | 188 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 356 | 187 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 187 | 98 | 89 | 52.41% | 52.41% | 52.41% | 2.41 pp | 9 | 16 | 0.56 |
| BTC Market Hours | nn | NN | 188 | 96 | 92 | 51.06% | 51.06% | 51.06% | 1.06 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 188 | 93 | 95 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 187 | 90 | 97 | 48.13% | 48.13% | 48.13% | 1.87 pp | -7 | 16 | -0.44 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | nn | NN | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 16 | -0.81 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Daily | mlp_sklearn | MLPClassifier | 190 | 90 | 100 | 47.37% | 47.37% | 47.37% | 2.63 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | rf | RandomForest | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 16 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 188 | 80 | 108 | 42.55% | 42.55% | 42.55% | 7.45 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 16 | -1.94 |
| BTC Market Hours | lstm | LSTM | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 187 | 75 | 112 | 40.11% | 40.11% | 40.11% | 9.89 pp | -37 | 16 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | nn | NN | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| BTC Daily | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 164 | 70 | 94 | 42.68% | 42.68% | 42.68% | 7.32 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 190 | 74 | 116 | 38.95% | 38.95% | 38.95% | 11.05 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 200 | 73 | 127 | 36.50% | 36.50% | 36.50% | 13.50 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 164 | 60 | 104 | 36.59% | 36.59% | 36.59% | 13.41 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 190 | 66 | 124 | 34.74% | 34.74% | 34.74% | 15.26 pp | -58 | 9 | -6.44 |
| BTC Hourly | xgb | XGBoost | 164 | 57 | 107 | 34.76% | 34.76% | 34.76% | 15.24 pp | -50 | 7 | -7.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 164 | 84 | 80 | 51.22% | 51.22% | 51.22% | 1.22 pp | 4 | 7 | 0.57 |
| BTC Hourly | transformer | Transformer | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 164 | 70 | 94 | 42.68% | 42.68% | 42.68% | 7.32 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 164 | 60 | 104 | 36.59% | 36.59% | 36.59% | 13.41 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 164 | 57 | 107 | 34.76% | 34.76% | 34.76% | 15.24 pp | -50 | 7 | -7.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 190 | 90 | 100 | 47.37% | 47.37% | 47.37% | 2.63 pp | -10 | 9 | -1.11 |
| BTC Daily | nn | NN | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| BTC Daily | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 190 | 74 | 116 | 38.95% | 38.95% | 38.95% | 11.05 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 200 | 73 | 127 | 36.50% | 36.50% | 36.50% | 13.50 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 190 | 66 | 124 | 34.74% | 34.74% | 34.74% | 15.26 pp | -58 | 9 | -6.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 188 | 96 | 92 | 51.06% | 51.06% | 51.06% | 1.06 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 188 | 93 | 95 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 188 | 80 | 108 | 42.55% | 42.55% | 42.55% | 7.45 pp | -28 | 15 | -1.87 |
| BTC Market Hours | lstm | LSTM | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 15 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 187 | 98 | 89 | 52.41% | 52.41% | 52.41% | 2.41 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 187 | 90 | 97 | 48.13% | 48.13% | 48.13% | 1.87 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 16 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 16 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 187 | 75 | 112 | 40.11% | 40.11% | 40.11% | 9.89 pp | -37 | 16 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
