# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T05:01:36.566361+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 257 | 197 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 354 | 185 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 354 | 185 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 158 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 185 | 97 | 88 | 52.43% | 52.43% | 52.43% | 2.43 pp | 9 | 16 | 0.56 |
| BTC Market Hours | nn | NN | 185 | 95 | 90 | 51.35% | 51.35% | 51.35% | 1.35 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 16 | -0.31 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 185 | 87 | 98 | 47.03% | 47.03% | 47.03% | 2.97 pp | -11 | 16 | -0.69 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| BTC Hourly | transformer | Transformer | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 9 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 16 | -1.31 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 16 | -1.81 |
| Consolidated Hourly | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 185 | 75 | 110 | 40.54% | 40.54% | 40.54% | 9.46 pp | -35 | 16 | -2.19 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Daily | nn | NN | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 9 | -2.78 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 161 | 68 | 93 | 42.24% | 42.24% | 42.24% | 7.76 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 7 | -3.86 |
| BTC Daily | rf | RandomForest | 187 | 74 | 113 | 39.57% | 39.57% | 39.57% | 10.43 pp | -39 | 9 | -4.33 |
| BTC Daily | xgb | XGBoost | 197 | 72 | 125 | 36.55% | 36.55% | 36.55% | 13.45 pp | -53 | 10 | -5.30 |
| BTC Daily | lstm | LSTM | 187 | 66 | 121 | 35.29% | 35.29% | 35.29% | 14.71 pp | -55 | 9 | -6.11 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 9 | -1.22 |
| BTC Daily | nn | NN | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 9 | -2.33 |
| BTC Daily | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 9 | -2.78 |
| BTC Daily | rf | RandomForest | 187 | 74 | 113 | 39.57% | 39.57% | 39.57% | 10.43 pp | -39 | 9 | -4.33 |
| BTC Daily | xgb | XGBoost | 197 | 72 | 125 | 36.55% | 36.55% | 36.55% | 13.45 pp | -53 | 10 | -5.30 |
| BTC Daily | lstm | LSTM | 187 | 66 | 121 | 35.29% | 35.29% | 35.29% | 14.71 pp | -55 | 9 | -6.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 185 | 95 | 90 | 51.35% | 51.35% | 51.35% | 1.35 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 15 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 15 | -0.87 |
| BTC Market Hours | xgb | XGBoost | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 15 | -1.80 |
| BTC Market Hours | lstm | LSTM | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 15 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 185 | 97 | 88 | 52.43% | 52.43% | 52.43% | 2.43 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 16 | -0.31 |
| BTC Market Hours Daily | nn | NN | 185 | 87 | 98 | 47.03% | 47.03% | 47.03% | 2.97 pp | -11 | 16 | -0.69 |
| BTC Market Hours Daily | rf | RandomForest | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 16 | -1.81 |
| BTC Market Hours Daily | lstm | LSTM | 185 | 75 | 110 | 40.54% | 40.54% | 40.54% | 9.46 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |

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
