# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T14:39:40.360640+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 163 | 103 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 199 | 139 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 246 | 127 | 119 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 246 | 127 | 119 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| BTC Market Hours | nn | NN | 127 | 68 | 59 | 53.54% | 53.54% | 53.54% | 3.54 pp | 9 | 10 | 0.90 |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 129 | 64 | 65 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 11 | -0.45 |
| BTC Market Hours | rf | RandomForest | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 11 | -0.82 |
| BTC Hourly | nn | NN | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| BTC Daily | nn | NN | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 127 | 55 | 72 | 43.31% | 43.31% | 43.31% | 6.69 pp | -17 | 10 | -1.70 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 127 | 50 | 77 | 39.37% | 39.37% | 39.37% | 10.63 pp | -27 | 11 | -2.45 |
| BTC Market Hours | xgb | XGBoost | 127 | 51 | 76 | 40.16% | 40.16% | 40.16% | 9.84 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 11 | -3.00 |
| BTC Market Hours | lstm | LSTM | 127 | 48 | 79 | 37.80% | 37.80% | 37.80% | 12.20 pp | -31 | 10 | -3.10 |
| BTC Hourly | rf | RandomForest | 103 | 43 | 60 | 41.75% | 41.75% | 41.75% | 8.25 pp | -17 | 5 | -3.40 |
| BTC Daily | rf | RandomForest | 129 | 54 | 75 | 41.86% | 41.86% | 41.86% | 8.14 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 7 | -5.29 |
| BTC Hourly | xgb | XGBoost | 103 | 36 | 67 | 34.95% | 34.95% | 34.95% | 15.05 pp | -31 | 5 | -6.20 |
| BTC Daily | lstm | LSTM | 129 | 45 | 84 | 34.88% | 34.88% | 34.88% | 15.12 pp | -39 | 6 | -6.50 |
| BTC Hourly | lstm | LSTM | 103 | 32 | 71 | 31.07% | 31.07% | 31.07% | 18.93 pp | -39 | 5 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 5 | -0.60 |
| BTC Hourly | nn | NN | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Hourly | rf | RandomForest | 103 | 43 | 60 | 41.75% | 41.75% | 41.75% | 8.25 pp | -17 | 5 | -3.40 |
| BTC Hourly | xgb | XGBoost | 103 | 36 | 67 | 34.95% | 34.95% | 34.95% | 15.05 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 103 | 32 | 71 | 31.07% | 31.07% | 31.07% | 18.93 pp | -39 | 5 | -7.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 129 | 64 | 65 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| BTC Daily | nn | NN | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 6 | -2.17 |
| BTC Daily | rf | RandomForest | 129 | 54 | 75 | 41.86% | 41.86% | 41.86% | 8.14 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 129 | 45 | 84 | 34.88% | 34.88% | 34.88% | 15.12 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 127 | 68 | 59 | 53.54% | 53.54% | 53.54% | 3.54 pp | 9 | 10 | 0.90 |
| BTC Market Hours | rf | RandomForest | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| BTC Market Hours | transformer | Transformer | 127 | 55 | 72 | 43.31% | 43.31% | 43.31% | 6.69 pp | -17 | 10 | -1.70 |
| BTC Market Hours | xgb | XGBoost | 127 | 51 | 76 | 40.16% | 40.16% | 40.16% | 9.84 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 127 | 48 | 79 | 37.80% | 37.80% | 37.80% | 12.20 pp | -31 | 10 | -3.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | transformer | Transformer | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | xgb | XGBoost | 127 | 50 | 77 | 39.37% | 39.37% | 39.37% | 10.63 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 11 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
