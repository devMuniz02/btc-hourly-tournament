# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T08:31:48.979366+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 106 | 46 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 142 | 82 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 148 | 70 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 148 | 70 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 1 | 54 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 1 | 54 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 70 | 40 | 30 | 57.14% | 57.14% | 57.14% | 7.14 pp | 10 | 6 | 1.67 |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 35 | 35 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Daily | transformer | Transformer | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| BTC Hourly | nn | NN | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 2 | -2.00 |
| BTC Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 4 | -3.50 |
| BTC Hourly | lstm | LSTM | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 2 | -4.00 |
| BTC Daily | lstm | LSTM | 72 | 26 | 46 | 36.11% | 36.11% | 36.11% | 13.89 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 82 | 26 | 56 | 31.71% | 31.71% | 31.71% | 18.29 pp | -30 | 5 | -6.00 |
| BTC Hourly | rf | RandomForest | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 2 | -7.00 |
| BTC Hourly | xgb | XGBoost | 46 | 14 | 32 | 30.43% | 30.43% | 30.43% | 19.57 pp | -18 | 2 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 2 | -1.00 |
| BTC Hourly | nn | NN | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 2 | -2.00 |
| BTC Hourly | lstm | LSTM | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 2 | -7.00 |
| BTC Hourly | xgb | XGBoost | 46 | 14 | 32 | 30.43% | 30.43% | 30.43% | 19.57 pp | -18 | 2 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Daily | transformer | Transformer | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 4 | -0.50 |
| BTC Daily | rf | RandomForest | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 72 | 26 | 46 | 36.11% | 36.11% | 36.11% | 13.89 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 82 | 26 | 56 | 31.71% | 31.71% | 31.71% | 18.29 pp | -30 | 5 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 70 | 40 | 30 | 57.14% | 57.14% | 57.14% | 7.14 pp | 10 | 6 | 1.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 70 | 33 | 37 | 47.14% | 47.14% | 47.14% | 2.86 pp | -4 | 6 | -0.67 |
| BTC Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 35 | 35 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | nn | NN | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | xgb | XGBoost | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
