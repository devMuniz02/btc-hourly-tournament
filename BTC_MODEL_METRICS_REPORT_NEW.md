# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T11:05:32.351190+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 161 | 101 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 197 | 137 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 242 | 125 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 242 | 125 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T21:00:00+00:00 | 103 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T21:00:00+00:00 | 103 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T21:00:00+00:00 | 103 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T21:00:00+00:00 | 104 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 11 | -0.45 |
| BTC Market Hours | rf | RandomForest | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 6 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 125 | 48 | 77 | 38.40% | 38.40% | 38.40% | 11.60 pp | -29 | 11 | -2.64 |
| BTC Market Hours Daily | lstm | LSTM | 125 | 46 | 79 | 36.80% | 36.80% | 36.80% | 13.20 pp | -33 | 11 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| BTC Market Hours | lstm | LSTM | 125 | 47 | 78 | 37.60% | 37.60% | 37.60% | 12.40 pp | -31 | 10 | -3.10 |
| BTC Daily | rf | RandomForest | 127 | 53 | 74 | 41.73% | 41.73% | 41.73% | 8.27 pp | -21 | 6 | -3.50 |
| BTC Hourly | rf | RandomForest | 101 | 41 | 60 | 40.59% | 40.59% | 40.59% | 9.41 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 137 | 50 | 87 | 36.50% | 36.50% | 36.50% | 13.50 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 127 | 44 | 83 | 34.65% | 34.65% | 34.65% | 15.35 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 101 | 34 | 67 | 33.66% | 33.66% | 33.66% | 16.34 pp | -33 | 5 | -6.60 |
| BTC Hourly | lstm | LSTM | 101 | 31 | 70 | 30.69% | 30.69% | 30.69% | 19.31 pp | -39 | 5 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Hourly | rf | RandomForest | 101 | 41 | 60 | 40.59% | 40.59% | 40.59% | 9.41 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 101 | 34 | 67 | 33.66% | 33.66% | 33.66% | 16.34 pp | -33 | 5 | -6.60 |
| BTC Hourly | lstm | LSTM | 101 | 31 | 70 | 30.69% | 30.69% | 30.69% | 19.31 pp | -39 | 5 | -7.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| BTC Daily | nn | NN | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 6 | -2.17 |
| BTC Daily | rf | RandomForest | 127 | 53 | 74 | 41.73% | 41.73% | 41.73% | 8.27 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 137 | 50 | 87 | 36.50% | 36.50% | 36.50% | 13.50 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 127 | 44 | 83 | 34.65% | 34.65% | 34.65% | 15.35 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| BTC Market Hours | rf | RandomForest | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| BTC Market Hours | transformer | Transformer | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| BTC Market Hours | xgb | XGBoost | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 125 | 47 | 78 | 37.60% | 37.60% | 37.60% | 12.40 pp | -31 | 10 | -3.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | nn | NN | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 125 | 48 | 77 | 38.40% | 38.40% | 38.40% | 11.60 pp | -29 | 11 | -2.64 |
| BTC Market Hours Daily | lstm | LSTM | 125 | 46 | 79 | 36.80% | 36.80% | 36.80% | 13.20 pp | -33 | 11 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | lstm | LSTM | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
