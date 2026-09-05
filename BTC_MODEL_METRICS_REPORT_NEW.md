# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T04:10:44.735105+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 221 | 161 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 256 | 196 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 353 | 184 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 353 | 184 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 184 | 96 | 88 | 52.17% | 52.17% | 52.17% | 2.17 pp | 8 | 16 | 0.50 |
| BTC Market Hours | nn | NN | 184 | 94 | 90 | 51.09% | 51.09% | 51.09% | 1.09 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| BTC Market Hours | transformer | Transformer | 184 | 91 | 93 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 184 | 89 | 95 | 48.37% | 48.37% | 48.37% | 1.63 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Hourly | transformer | Transformer | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 184 | 81 | 103 | 44.02% | 44.02% | 44.02% | 5.98 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 184 | 77 | 107 | 41.85% | 41.85% | 41.85% | 8.15 pp | -30 | 16 | -1.88 |
| Consolidated Hourly | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 184 | 74 | 110 | 40.22% | 40.22% | 40.22% | 9.78 pp | -36 | 16 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Daily | nn | NN | 186 | 82 | 104 | 44.09% | 44.09% | 44.09% | 5.91 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 9 | -2.89 |
| BTC Hourly | nn | NN | 161 | 68 | 93 | 42.24% | 42.24% | 42.24% | 7.76 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 7 | -3.86 |
| BTC Daily | rf | RandomForest | 186 | 73 | 113 | 39.25% | 39.25% | 39.25% | 10.75 pp | -40 | 9 | -4.44 |
| BTC Daily | xgb | XGBoost | 196 | 71 | 125 | 36.22% | 36.22% | 36.22% | 13.78 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 186 | 66 | 120 | 35.48% | 35.48% | 35.48% | 14.52 pp | -54 | 9 | -6.00 |
| BTC Hourly | lstm | LSTM | 161 | 58 | 103 | 36.02% | 36.02% | 36.02% | 13.98 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 161 | 56 | 105 | 34.78% | 34.78% | 34.78% | 15.22 pp | -49 | 7 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 7 | -1.00 |
| BTC Hourly | nn | NN | 161 | 68 | 93 | 42.24% | 42.24% | 42.24% | 7.76 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 7 | -3.86 |
| BTC Hourly | lstm | LSTM | 161 | 58 | 103 | 36.02% | 36.02% | 36.02% | 13.98 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 161 | 56 | 105 | 34.78% | 34.78% | 34.78% | 15.22 pp | -49 | 7 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 9 | -1.33 |
| BTC Daily | nn | NN | 186 | 82 | 104 | 44.09% | 44.09% | 44.09% | 5.91 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 9 | -2.89 |
| BTC Daily | rf | RandomForest | 186 | 73 | 113 | 39.25% | 39.25% | 39.25% | 10.75 pp | -40 | 9 | -4.44 |
| BTC Daily | xgb | XGBoost | 196 | 71 | 125 | 36.22% | 36.22% | 36.22% | 13.78 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 186 | 66 | 120 | 35.48% | 35.48% | 35.48% | 14.52 pp | -54 | 9 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 184 | 94 | 90 | 51.09% | 51.09% | 51.09% | 1.09 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 184 | 91 | 93 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 184 | 96 | 88 | 52.17% | 52.17% | 52.17% | 2.17 pp | 8 | 16 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 184 | 89 | 95 | 48.37% | 48.37% | 48.37% | 1.63 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 184 | 81 | 103 | 44.02% | 44.02% | 44.02% | 5.98 pp | -22 | 16 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 184 | 77 | 107 | 41.85% | 41.85% | 41.85% | 8.15 pp | -30 | 16 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 184 | 74 | 110 | 40.22% | 40.22% | 40.22% | 9.78 pp | -36 | 16 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
