# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T06:01:16.210667+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 222 | 162 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 258 | 198 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 355 | 186 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 355 | 186 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 162 | 84 | 78 | 51.85% | 51.85% | 51.85% | 1.85 pp | 6 | 7 | 0.86 |
| BTC Market Hours Daily | transformer | Transformer | 186 | 98 | 88 | 52.69% | 52.69% | 52.69% | 2.69 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 186 | 95 | 91 | 51.08% | 51.08% | 51.08% | 1.08 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| BTC Market Hours | transformer | Transformer | 186 | 92 | 94 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 186 | 90 | 96 | 48.39% | 48.39% | 48.39% | 1.61 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 162 | 78 | 84 | 48.15% | 48.15% | 48.15% | 1.85 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 15 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 186 | 83 | 103 | 44.62% | 44.62% | 44.62% | 5.38 pp | -20 | 16 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 9 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 186 | 79 | 107 | 42.47% | 42.47% | 42.47% | 7.53 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 16 | -1.88 |
| BTC Market Hours | lstm | LSTM | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 186 | 75 | 111 | 40.32% | 40.32% | 40.32% | 9.68 pp | -36 | 16 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Daily | nn | NN | 188 | 83 | 105 | 44.15% | 44.15% | 44.15% | 5.85 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 188 | 82 | 106 | 43.62% | 43.62% | 43.62% | 6.38 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Hourly | nn | NN | 162 | 69 | 93 | 42.59% | 42.59% | 42.59% | 7.41 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 188 | 74 | 114 | 39.36% | 39.36% | 39.36% | 10.64 pp | -40 | 9 | -4.44 |
| BTC Daily | xgb | XGBoost | 198 | 73 | 125 | 36.87% | 36.87% | 36.87% | 13.13 pp | -52 | 10 | -5.20 |
| BTC Daily | lstm | LSTM | 188 | 66 | 122 | 35.11% | 35.11% | 35.11% | 14.89 pp | -56 | 9 | -6.22 |
| BTC Hourly | lstm | LSTM | 162 | 59 | 103 | 36.42% | 36.42% | 36.42% | 13.58 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 162 | 57 | 105 | 35.19% | 35.19% | 35.19% | 14.81 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 162 | 84 | 78 | 51.85% | 51.85% | 51.85% | 1.85 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 162 | 78 | 84 | 48.15% | 48.15% | 48.15% | 1.85 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 162 | 69 | 93 | 42.59% | 42.59% | 42.59% | 7.41 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 162 | 59 | 103 | 36.42% | 36.42% | 36.42% | 13.58 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 162 | 57 | 105 | 35.19% | 35.19% | 35.19% | 14.81 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 9 | -1.33 |
| BTC Daily | nn | NN | 188 | 83 | 105 | 44.15% | 44.15% | 44.15% | 5.85 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 188 | 82 | 106 | 43.62% | 43.62% | 43.62% | 6.38 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 188 | 74 | 114 | 39.36% | 39.36% | 39.36% | 10.64 pp | -40 | 9 | -4.44 |
| BTC Daily | xgb | XGBoost | 198 | 73 | 125 | 36.87% | 36.87% | 36.87% | 13.13 pp | -52 | 10 | -5.20 |
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
