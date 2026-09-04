# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T15:33:21.343800+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 212 | 152 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 248 | 188 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 14:00:00+00:00 | 335 | 176 | 159 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 14:00:00+00:00 | 335 | 176 | 159 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 150 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 150 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 150 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 151 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 7 | 0.86 |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 176 | 92 | 84 | 52.27% | 52.27% | 52.27% | 2.27 pp | 8 | 14 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 176 | 89 | 87 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 15 | 0.13 |
| Consolidated Hourly | rf | RandomForest | 150 | 75 | 75 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 150 | 75 | 75 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 176 | 86 | 90 | 48.86% | 48.86% | 48.86% | 1.14 pp | -4 | 15 | -0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 11 | -0.55 |
| BTC Hourly | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| BTC Market Hours | transformer | Transformer | 176 | 84 | 92 | 47.73% | 47.73% | 47.73% | 2.27 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 176 | 82 | 94 | 46.59% | 46.59% | 46.59% | 3.41 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 176 | 80 | 96 | 45.45% | 45.45% | 45.45% | 4.55 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 176 | 78 | 98 | 44.32% | 44.32% | 44.32% | 5.68 pp | -20 | 15 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | nn | NN | 150 | 65 | 85 | 43.33% | 43.33% | 43.33% | 6.67 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 150 | 65 | 85 | 43.33% | 43.33% | 43.33% | 6.67 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 176 | 75 | 101 | 42.61% | 42.61% | 42.61% | 7.39 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 15 | -2.00 |
| BTC Market Hours | lstm | LSTM | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | transformer | Transformer | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 11 | -2.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 176 | 70 | 106 | 39.77% | 39.77% | 39.77% | 10.23 pp | -36 | 15 | -2.40 |
| BTC Daily | nn | NN | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 8 | -2.75 |
| BTC Daily | transformer | Transformer | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 152 | 65 | 87 | 42.76% | 42.76% | 42.76% | 7.24 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 152 | 62 | 90 | 40.79% | 40.79% | 40.79% | 9.21 pp | -28 | 7 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 178 | 72 | 106 | 40.45% | 40.45% | 40.45% | 9.55 pp | -34 | 8 | -4.25 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 188 | 69 | 119 | 36.70% | 36.70% | 36.70% | 13.30 pp | -50 | 9 | -5.56 |
| BTC Hourly | lstm | LSTM | 152 | 55 | 97 | 36.18% | 36.18% | 36.18% | 13.82 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 178 | 62 | 116 | 34.83% | 34.83% | 34.83% | 15.17 pp | -54 | 8 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 152 | 65 | 87 | 42.76% | 42.76% | 42.76% | 7.24 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 152 | 62 | 90 | 40.79% | 40.79% | 40.79% | 9.21 pp | -28 | 7 | -4.00 |
| BTC Hourly | lstm | LSTM | 152 | 55 | 97 | 36.18% | 36.18% | 36.18% | 13.82 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 8 | -2.75 |
| BTC Daily | transformer | Transformer | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 178 | 72 | 106 | 40.45% | 40.45% | 40.45% | 9.55 pp | -34 | 8 | -4.25 |
| BTC Daily | xgb | XGBoost | 188 | 69 | 119 | 36.70% | 36.70% | 36.70% | 13.30 pp | -50 | 9 | -5.56 |
| BTC Daily | lstm | LSTM | 178 | 62 | 116 | 34.83% | 34.83% | 34.83% | 15.17 pp | -54 | 8 | -6.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 176 | 92 | 84 | 52.27% | 52.27% | 52.27% | 2.27 pp | 8 | 14 | 0.57 |
| BTC Market Hours | transformer | Transformer | 176 | 84 | 92 | 47.73% | 47.73% | 47.73% | 2.27 pp | -8 | 14 | -0.57 |
| BTC Market Hours | rf | RandomForest | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 14 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 176 | 80 | 96 | 45.45% | 45.45% | 45.45% | 4.55 pp | -16 | 14 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 176 | 75 | 101 | 42.61% | 42.61% | 42.61% | 7.39 pp | -26 | 14 | -1.86 |
| BTC Market Hours | lstm | LSTM | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 176 | 89 | 87 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 15 | 0.13 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 176 | 86 | 90 | 48.86% | 48.86% | 48.86% | 1.14 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 176 | 82 | 94 | 46.59% | 46.59% | 46.59% | 3.41 pp | -12 | 15 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 176 | 78 | 98 | 44.32% | 44.32% | 44.32% | 5.68 pp | -20 | 15 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 176 | 70 | 106 | 39.77% | 39.77% | 39.77% | 10.23 pp | -36 | 15 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 150 | 75 | 75 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | lstm | LSTM | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 150 | 65 | 85 | 43.33% | 43.33% | 43.33% | 6.67 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 11 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 150 | 75 | 75 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 150 | 65 | 85 | 43.33% | 43.33% | 43.33% | 6.67 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 11 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
