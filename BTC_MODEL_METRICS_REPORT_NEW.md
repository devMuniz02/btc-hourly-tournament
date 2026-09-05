# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T11:39:44.740207+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 226 | 166 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 262 | 202 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 359 | 190 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 359 | 190 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 163 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 190 | 99 | 91 | 52.11% | 52.11% | 52.11% | 2.11 pp | 8 | 16 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 190 | 92 | 98 | 48.42% | 48.42% | 48.42% | 1.58 pp | -6 | 16 | -0.38 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | nn | NN | 190 | 89 | 101 | 46.84% | 46.84% | 46.84% | 3.16 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 190 | 84 | 106 | 44.21% | 44.21% | 44.21% | 5.79 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 190 | 78 | 112 | 41.05% | 41.05% | 41.05% | 8.95 pp | -34 | 16 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 190 | 77 | 113 | 40.53% | 40.53% | 40.53% | 9.47 pp | -36 | 16 | -2.25 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 190 | 99 | 91 | 52.11% | 52.11% | 52.11% | 2.11 pp | 8 | 16 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 190 | 92 | 98 | 48.42% | 48.42% | 48.42% | 1.58 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 190 | 89 | 101 | 46.84% | 46.84% | 46.84% | 3.16 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 190 | 84 | 106 | 44.21% | 44.21% | 44.21% | 5.79 pp | -22 | 16 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 190 | 78 | 112 | 41.05% | 41.05% | 41.05% | 8.95 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 190 | 77 | 113 | 40.53% | 40.53% | 40.53% | 9.47 pp | -36 | 16 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
