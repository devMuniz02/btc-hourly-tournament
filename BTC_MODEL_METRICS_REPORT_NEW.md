# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T06:15:24.183970+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 104 | 44 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 140 | 80 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 146 | 68 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 146 | 68 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 14:00:00+00:00 | 53 | 53 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 14:00:00+00:00 | 53 | 53 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 14:00:00+00:00 | 53 | 1 | 52 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 14:00:00+00:00 | 53 | 1 | 52 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 68 | 38 | 30 | 55.88% | 55.88% | 55.88% | 5.88 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 6 | 0.83 |
| BTC Market Hours Daily | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| BTC Daily | transformer | Transformer | 70 | 35 | 35 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 6 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | nn | NN | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |
| BTC Hourly | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 80 | 26 | 54 | 32.50% | 32.50% | 32.50% | 17.50 pp | -28 | 5 | -5.60 |
| BTC Hourly | rf | RandomForest | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 2 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 2 | -3.00 |
| BTC Hourly | rf | RandomForest | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 2 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 70 | 35 | 35 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 4 | -0.50 |
| BTC Daily | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 80 | 26 | 54 | 32.50% | 32.50% | 32.50% | 17.50 pp | -28 | 5 | -5.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 68 | 38 | 30 | 55.88% | 55.88% | 55.88% | 5.88 pp | 8 | 6 | 1.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 6 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | nn | NN | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
