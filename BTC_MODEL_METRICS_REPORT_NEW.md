# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T06:45:14.569472+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 123 | 63 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 159 | 99 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 178 | 87 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 178 | 87 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 1 | 69 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 1 | 69 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 87 | 49 | 38 | 56.32% | 56.32% | 56.32% | 6.32 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| BTC Hourly | nn | NN | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 8 | -0.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 5 | -1.00 |
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
| BTC Market Hours Daily | rf | RandomForest | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 8 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |
| BTC Market Hours Daily | xgb | XGBoost | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 8 | -2.88 |
| BTC Market Hours | xgb | XGBoost | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 5 | -4.60 |
| BTC Daily | lstm | LSTM | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 5 | -5.00 |
| BTC Hourly | lstm | LSTM | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 99 | 30 | 69 | 30.30% | 30.30% | 30.30% | 19.70 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 63 | 21 | 42 | 33.33% | 33.33% | 33.33% | 16.67 pp | -21 | 3 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 3 | -1.00 |
| BTC Hourly | rf | RandomForest | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 3 | -5.67 |
| BTC Hourly | xgb | XGBoost | 63 | 21 | 42 | 33.33% | 33.33% | 33.33% | 16.67 pp | -21 | 3 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 5 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | rf | RandomForest | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 5 | -4.60 |
| BTC Daily | lstm | LSTM | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 99 | 30 | 69 | 30.30% | 30.30% | 30.30% | 19.70 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 87 | 49 | 38 | 56.32% | 56.32% | 56.32% | 6.32 pp | 11 | 7 | 1.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | transformer | Transformer | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 8 | -2.88 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |

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
