# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T02:30:22.294148+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 236 | 176 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 272 | 212 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 382 | 200 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 382 | 200 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 173 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 200 | 103 | 97 | 51.50% | 51.50% | 51.50% | 1.50 pp | 6 | 17 | 0.35 |
| BTC Market Hours | nn | NN | 200 | 102 | 98 | 51.00% | 51.00% | 51.00% | 1.00 pp | 4 | 16 | 0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 176 | 89 | 87 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 8 | 0.25 |
| BTC Market Hours | transformer | Transformer | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 16 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 200 | 97 | 103 | 48.50% | 48.50% | 48.50% | 1.50 pp | -6 | 17 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 200 | 95 | 105 | 47.50% | 47.50% | 47.50% | 2.50 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 200 | 94 | 106 | 47.00% | 47.00% | 47.00% | 3.00 pp | -12 | 16 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 17 | -1.53 |
| Consolidated Hourly | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| BTC Hourly | transformer | Transformer | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 8 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 200 | 86 | 114 | 43.00% | 43.00% | 43.00% | 7.00 pp | -28 | 16 | -1.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| BTC Daily | nn | NN | 202 | 92 | 110 | 45.54% | 45.54% | 45.54% | 4.46 pp | -18 | 9 | -2.00 |
| BTC Market Hours | lstm | LSTM | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 16 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 200 | 82 | 118 | 41.00% | 41.00% | 41.00% | 9.00 pp | -36 | 17 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 200 | 81 | 119 | 40.50% | 40.50% | 40.50% | 9.50 pp | -38 | 17 | -2.24 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Daily | transformer | Transformer | 202 | 86 | 116 | 42.57% | 42.57% | 42.57% | 7.43 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 176 | 74 | 102 | 42.05% | 42.05% | 42.05% | 7.95 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Hourly | nn | NN | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 8 | -3.75 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 202 | 77 | 125 | 38.12% | 38.12% | 38.12% | 11.88 pp | -48 | 9 | -5.33 |
| BTC Hourly | lstm | LSTM | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 212 | 76 | 136 | 35.85% | 35.85% | 35.85% | 14.15 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 202 | 68 | 134 | 33.66% | 33.66% | 33.66% | 16.34 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 176 | 89 | 87 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 8 | -1.75 |
| BTC Hourly | rf | RandomForest | 176 | 74 | 102 | 42.05% | 42.05% | 42.05% | 7.95 pp | -28 | 8 | -3.50 |
| BTC Hourly | nn | NN | 176 | 73 | 103 | 41.48% | 41.48% | 41.48% | 8.52 pp | -30 | 8 | -3.75 |
| BTC Hourly | lstm | LSTM | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 176 | 64 | 112 | 36.36% | 36.36% | 36.36% | 13.64 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 202 | 92 | 110 | 45.54% | 45.54% | 45.54% | 4.46 pp | -18 | 9 | -2.00 |
| BTC Daily | transformer | Transformer | 202 | 86 | 116 | 42.57% | 42.57% | 42.57% | 7.43 pp | -30 | 9 | -3.33 |
| BTC Daily | rf | RandomForest | 202 | 77 | 125 | 38.12% | 38.12% | 38.12% | 11.88 pp | -48 | 9 | -5.33 |
| BTC Daily | xgb | XGBoost | 212 | 76 | 136 | 35.85% | 35.85% | 35.85% | 14.15 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 202 | 68 | 134 | 33.66% | 33.66% | 33.66% | 16.34 pp | -66 | 9 | -7.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 200 | 102 | 98 | 51.00% | 51.00% | 51.00% | 1.00 pp | 4 | 16 | 0.25 |
| BTC Market Hours | transformer | Transformer | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 200 | 94 | 106 | 47.00% | 47.00% | 47.00% | 3.00 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 200 | 86 | 114 | 43.00% | 43.00% | 43.00% | 7.00 pp | -28 | 16 | -1.75 |
| BTC Market Hours | lstm | LSTM | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 16 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 200 | 103 | 97 | 51.50% | 51.50% | 51.50% | 1.50 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 200 | 97 | 103 | 48.50% | 48.50% | 48.50% | 1.50 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 200 | 95 | 105 | 47.50% | 47.50% | 47.50% | 2.50 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 200 | 82 | 118 | 41.00% | 41.00% | 41.00% | 9.00 pp | -36 | 17 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 200 | 81 | 119 | 40.50% | 40.50% | 40.50% | 9.50 pp | -38 | 17 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
