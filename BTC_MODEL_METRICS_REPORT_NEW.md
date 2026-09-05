# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T07:06:24.751533+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 223 | 163 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 258 | 198 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 355 | 186 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 355 | 186 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 163 | 84 | 79 | 51.53% | 51.53% | 51.53% | 1.53 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 186 | 98 | 88 | 52.69% | 52.69% | 52.69% | 2.69 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 186 | 95 | 91 | 51.08% | 51.08% | 51.08% | 1.08 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 186 | 92 | 94 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 186 | 90 | 96 | 48.39% | 48.39% | 48.39% | 1.61 pp | -6 | 16 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 16 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 186 | 83 | 103 | 44.62% | 44.62% | 44.62% | 5.38 pp | -20 | 16 | -1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 9 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 186 | 79 | 107 | 42.47% | 42.47% | 42.47% | 7.53 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 16 | -1.88 |
| BTC Market Hours | lstm | LSTM | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 186 | 75 | 111 | 40.32% | 40.32% | 40.32% | 9.68 pp | -36 | 16 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | nn | NN | 188 | 82 | 106 | 43.62% | 43.62% | 43.62% | 6.38 pp | -24 | 9 | -2.67 |
| BTC Daily | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 9 | -2.89 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 188 | 73 | 115 | 38.83% | 38.83% | 38.83% | 11.17 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 198 | 72 | 126 | 36.36% | 36.36% | 36.36% | 13.64 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 163 | 60 | 103 | 36.81% | 36.81% | 36.81% | 13.19 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 188 | 66 | 122 | 35.11% | 35.11% | 35.11% | 14.89 pp | -56 | 9 | -6.22 |
| BTC Hourly | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 7 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 163 | 84 | 79 | 51.53% | 51.53% | 51.53% | 1.53 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 163 | 60 | 103 | 36.81% | 36.81% | 36.81% | 13.19 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 7 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 9 | -1.33 |
| BTC Daily | nn | NN | 188 | 82 | 106 | 43.62% | 43.62% | 43.62% | 6.38 pp | -24 | 9 | -2.67 |
| BTC Daily | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 9 | -2.89 |
| BTC Daily | rf | RandomForest | 188 | 73 | 115 | 38.83% | 38.83% | 38.83% | 11.17 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 198 | 72 | 126 | 36.36% | 36.36% | 36.36% | 13.64 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 188 | 66 | 122 | 35.11% | 35.11% | 35.11% | 14.89 pp | -56 | 9 | -6.22 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 186 | 95 | 91 | 51.08% | 51.08% | 51.08% | 1.08 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 186 | 92 | 94 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 186 | 79 | 107 | 42.47% | 42.47% | 42.47% | 7.53 pp | -28 | 15 | -1.87 |
| BTC Market Hours | lstm | LSTM | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 15 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 186 | 98 | 88 | 52.69% | 52.69% | 52.69% | 2.69 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 186 | 90 | 96 | 48.39% | 48.39% | 48.39% | 1.61 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 186 | 83 | 103 | 44.62% | 44.62% | 44.62% | 5.38 pp | -20 | 16 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 16 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 186 | 75 | 111 | 40.32% | 40.32% | 40.32% | 9.68 pp | -36 | 16 | -2.25 |

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
