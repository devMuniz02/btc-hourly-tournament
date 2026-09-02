# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T15:55:02.413702+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 180 | 120 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 216 | 156 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 14:00:00+00:00 | 277 | 144 | 133 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 14:00:00+00:00 | 277 | 144 | 133 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 120 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 120 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 120 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 121 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Market Hours | nn | NN | 144 | 75 | 69 | 52.08% | 52.08% | 52.08% | 2.08 pp | 6 | 12 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 120 | 61 | 59 | 50.83% | 50.83% | 50.83% | 0.83 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 120 | 61 | 59 | 50.83% | 50.83% | 50.83% | 0.83 pp | 2 | 10 | 0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 146 | 72 | 74 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| BTC Hourly | transformer | Transformer | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 144 | 69 | 75 | 47.92% | 47.92% | 47.92% | 2.08 pp | -6 | 12 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 144 | 68 | 76 | 47.22% | 47.22% | 47.22% | 2.78 pp | -8 | 12 | -0.67 |
| BTC Market Hours | rf | RandomForest | 144 | 67 | 77 | 46.53% | 46.53% | 46.53% | 3.47 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 144 | 66 | 78 | 45.83% | 45.83% | 45.83% | 4.17 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours | transformer | Transformer | 144 | 65 | 79 | 45.14% | 45.14% | 45.14% | 4.86 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | nn | NN | 120 | 53 | 67 | 44.17% | 44.17% | 44.17% | 5.83 pp | -14 | 10 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 120 | 53 | 67 | 44.17% | 44.17% | 44.17% | 5.83 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 120 | 56 | 64 | 46.67% | 46.67% | 46.67% | 3.33 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 120 | 52 | 68 | 43.33% | 43.33% | 43.33% | 6.67 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 120 | 52 | 68 | 43.33% | 43.33% | 43.33% | 6.67 pp | -16 | 10 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| BTC Daily | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 144 | 58 | 86 | 40.28% | 40.28% | 40.28% | 9.72 pp | -28 | 12 | -2.33 |
| BTC Market Hours | lstm | LSTM | 144 | 57 | 87 | 39.58% | 39.58% | 39.58% | 10.42 pp | -30 | 12 | -2.50 |
| BTC Daily | transformer | Transformer | 146 | 64 | 82 | 43.84% | 43.84% | 43.84% | 6.16 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 144 | 54 | 90 | 37.50% | 37.50% | 37.50% | 12.50 pp | -36 | 12 | -3.00 |
| BTC Daily | rf | RandomForest | 146 | 62 | 84 | 42.47% | 42.47% | 42.47% | 7.53 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 120 | 50 | 70 | 41.67% | 41.67% | 41.67% | 8.33 pp | -20 | 5 | -4.00 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 156 | 58 | 98 | 37.18% | 37.18% | 37.18% | 12.82 pp | -40 | 8 | -5.00 |
| BTC Hourly | xgb | XGBoost | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 5 | -6.00 |
| BTC Daily | lstm | LSTM | 146 | 52 | 94 | 35.62% | 35.62% | 35.62% | 14.38 pp | -42 | 7 | -6.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| BTC Hourly | lstm | LSTM | 120 | 40 | 80 | 33.33% | 33.33% | 33.33% | 16.67 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | transformer | Transformer | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 120 | 56 | 64 | 46.67% | 46.67% | 46.67% | 3.33 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 120 | 50 | 70 | 41.67% | 41.67% | 41.67% | 8.33 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 5 | -6.00 |
| BTC Hourly | lstm | LSTM | 120 | 40 | 80 | 33.33% | 33.33% | 33.33% | 16.67 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 146 | 72 | 74 | 49.32% | 49.32% | 49.32% | 0.68 pp | -2 | 7 | -0.29 |
| BTC Daily | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 146 | 64 | 82 | 43.84% | 43.84% | 43.84% | 6.16 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 146 | 62 | 84 | 42.47% | 42.47% | 42.47% | 7.53 pp | -22 | 7 | -3.14 |
| BTC Daily | xgb | XGBoost | 156 | 58 | 98 | 37.18% | 37.18% | 37.18% | 12.82 pp | -40 | 8 | -5.00 |
| BTC Daily | lstm | LSTM | 146 | 52 | 94 | 35.62% | 35.62% | 35.62% | 14.38 pp | -42 | 7 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 144 | 75 | 69 | 52.08% | 52.08% | 52.08% | 2.08 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 144 | 67 | 77 | 46.53% | 46.53% | 46.53% | 3.47 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 144 | 66 | 78 | 45.83% | 45.83% | 45.83% | 4.17 pp | -12 | 12 | -1.00 |
| BTC Market Hours | transformer | Transformer | 144 | 65 | 79 | 45.14% | 45.14% | 45.14% | 4.86 pp | -14 | 12 | -1.17 |
| BTC Market Hours | xgb | XGBoost | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 12 | -2.00 |
| BTC Market Hours | lstm | LSTM | 144 | 57 | 87 | 39.58% | 39.58% | 39.58% | 10.42 pp | -30 | 12 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 144 | 69 | 75 | 47.92% | 47.92% | 47.92% | 2.08 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 144 | 68 | 76 | 47.22% | 47.22% | 47.22% | 2.78 pp | -8 | 12 | -0.67 |
| BTC Market Hours Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 12 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 144 | 58 | 86 | 40.28% | 40.28% | 40.28% | 9.72 pp | -28 | 12 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 144 | 54 | 90 | 37.50% | 37.50% | 37.50% | 12.50 pp | -36 | 12 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 120 | 61 | 59 | 50.83% | 50.83% | 50.83% | 0.83 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 120 | 53 | 67 | 44.17% | 44.17% | 44.17% | 5.83 pp | -14 | 10 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 120 | 52 | 68 | 43.33% | 43.33% | 43.33% | 6.67 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 120 | 61 | 59 | 50.83% | 50.83% | 50.83% | 0.83 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 120 | 58 | 62 | 48.33% | 48.33% | 48.33% | 1.67 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 120 | 53 | 67 | 44.17% | 44.17% | 44.17% | 5.83 pp | -14 | 10 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 120 | 52 | 68 | 43.33% | 43.33% | 43.33% | 6.67 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
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
