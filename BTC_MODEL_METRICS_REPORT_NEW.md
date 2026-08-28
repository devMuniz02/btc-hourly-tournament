# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T14:39:49.989459+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 32 | 70 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 128 | 68 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 13:00:00+00:00 | 123 | 56 | 67 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 13:00:00+00:00 | 123 | 56 | 67 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 15:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 15:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 15:00:00+00:00 | 43 | 1 | 42 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 15:00:00+00:00 | 43 | 1 | 42 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| BTC Market Hours | nn | NN | 56 | 31 | 25 | 55.36% | 55.36% | 55.36% | 5.36 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | nn | NN | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours | rf | RandomForest | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Hourly | lstm | LSTM | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Hourly | transformer | Transformer | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Market Hours | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 5 | -3.60 |
| BTC Market Hours Daily | lstm | LSTM | 56 | 17 | 39 | 30.36% | 30.36% | 30.36% | 19.64 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 68 | 23 | 45 | 33.82% | 33.82% | 33.82% | 16.18 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 58 | 20 | 38 | 34.48% | 34.48% | 34.48% | 15.52 pp | -18 | 3 | -6.00 |
| BTC Hourly | rf | RandomForest | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 2 | 1.00 |
| BTC Hourly | nn | NN | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Hourly | transformer | Transformer | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 2 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 68 | 23 | 45 | 33.82% | 33.82% | 33.82% | 16.18 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 58 | 20 | 38 | 34.48% | 34.48% | 34.48% | 15.52 pp | -18 | 3 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 56 | 31 | 25 | 55.36% | 55.36% | 55.36% | 5.36 pp | 6 | 5 | 1.20 |
| BTC Market Hours | rf | RandomForest | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 5 | -3.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 56 | 17 | 39 | 30.36% | 30.36% | 30.36% | 19.64 pp | -22 | 6 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
