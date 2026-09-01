# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T07:30:56.892202+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 159 | 99 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 194 | 134 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 239 | 122 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 239 | 122 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| BTC Market Hours | nn | NN | 122 | 64 | 58 | 52.46% | 52.46% | 52.46% | 2.46 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 122 | 60 | 62 | 49.18% | 49.18% | 49.18% | 0.82 pp | -2 | 11 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| BTC Market Hours | rf | RandomForest | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | transformer | Transformer | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 11 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | nn | NN | 122 | 54 | 68 | 44.26% | 44.26% | 44.26% | 5.74 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| BTC Daily | transformer | Transformer | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 122 | 48 | 74 | 39.34% | 39.34% | 39.34% | 10.66 pp | -26 | 11 | -2.36 |
| BTC Market Hours | xgb | XGBoost | 122 | 49 | 73 | 40.16% | 40.16% | 40.16% | 9.84 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 122 | 46 | 76 | 37.70% | 37.70% | 37.70% | 12.30 pp | -30 | 11 | -2.73 |
| BTC Market Hours | lstm | LSTM | 122 | 47 | 75 | 38.52% | 38.52% | 38.52% | 11.48 pp | -28 | 10 | -2.80 |
| BTC Daily | rf | RandomForest | 124 | 51 | 73 | 41.13% | 41.13% | 41.13% | 8.87 pp | -22 | 6 | -3.67 |
| BTC Hourly | rf | RandomForest | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 134 | 48 | 86 | 35.82% | 35.82% | 35.82% | 14.18 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 124 | 43 | 81 | 34.68% | 34.68% | 34.68% | 15.32 pp | -38 | 6 | -6.33 |
| BTC Hourly | xgb | XGBoost | 99 | 32 | 67 | 32.32% | 32.32% | 32.32% | 17.68 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 99 | 31 | 68 | 31.31% | 31.31% | 31.31% | 18.69 pp | -37 | 5 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | rf | RandomForest | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 99 | 32 | 67 | 32.32% | 32.32% | 32.32% | 17.68 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 99 | 31 | 68 | 31.31% | 31.31% | 31.31% | 18.69 pp | -37 | 5 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Daily | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 124 | 51 | 73 | 41.13% | 41.13% | 41.13% | 8.87 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 134 | 48 | 86 | 35.82% | 35.82% | 35.82% | 14.18 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 124 | 43 | 81 | 34.68% | 34.68% | 34.68% | 15.32 pp | -38 | 6 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 122 | 64 | 58 | 52.46% | 52.46% | 52.46% | 2.46 pp | 6 | 10 | 0.60 |
| BTC Market Hours | rf | RandomForest | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 10 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 10 | -1.00 |
| BTC Market Hours | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 122 | 49 | 73 | 40.16% | 40.16% | 40.16% | 9.84 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 122 | 47 | 75 | 38.52% | 38.52% | 38.52% | 11.48 pp | -28 | 10 | -2.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 122 | 60 | 62 | 49.18% | 49.18% | 49.18% | 0.82 pp | -2 | 11 | -0.18 |
| BTC Market Hours Daily | rf | RandomForest | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | transformer | Transformer | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | nn | NN | 122 | 54 | 68 | 44.26% | 44.26% | 44.26% | 5.74 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 122 | 48 | 74 | 39.34% | 39.34% | 39.34% | 10.66 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 122 | 46 | 76 | 37.70% | 37.70% | 37.70% | 12.30 pp | -30 | 11 | -2.73 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
