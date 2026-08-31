# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T02:35:50.840912+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 139 | 79 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 175 | 115 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 207 | 103 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 207 | 103 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 1 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 1 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 103 | 56 | 47 | 54.37% | 54.37% | 54.37% | 4.37 pp | 9 | 8 | 1.12 |
| Consolidated Hourly | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| BTC Hourly | transformer | Transformer | 79 | 41 | 38 | 51.90% | 51.90% | 51.90% | 1.90 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| BTC Hourly | nn | NN | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 8 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Daily | nn | NN | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| Consolidated Hourly | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 4 | -2.25 |
| BTC Market Hours | lstm | LSTM | 103 | 42 | 61 | 40.78% | 40.78% | 40.78% | 9.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 103 | 40 | 63 | 38.83% | 38.83% | 38.83% | 11.17 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 105 | 46 | 59 | 43.81% | 43.81% | 43.81% | 6.19 pp | -13 | 5 | -2.60 |
| BTC Market Hours | transformer | Transformer | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | xgb | XGBoost | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 9 | -3.00 |
| BTC Market Hours | xgb | XGBoost | 103 | 39 | 64 | 37.86% | 37.86% | 37.86% | 12.14 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 105 | 43 | 62 | 40.95% | 40.95% | 40.95% | 9.05 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 115 | 41 | 74 | 35.65% | 35.65% | 35.65% | 14.35 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 105 | 38 | 67 | 36.19% | 36.19% | 36.19% | 13.81 pp | -29 | 5 | -5.80 |
| BTC Hourly | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 79 | 41 | 38 | 51.90% | 51.90% | 51.90% | 1.90 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 4 | -2.25 |
| BTC Hourly | rf | RandomForest | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 4 | -3.25 |
| BTC Hourly | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 105 | 46 | 59 | 43.81% | 43.81% | 43.81% | 6.19 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 105 | 43 | 62 | 40.95% | 40.95% | 40.95% | 9.05 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 115 | 41 | 74 | 35.65% | 35.65% | 35.65% | 14.35 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 105 | 38 | 67 | 36.19% | 36.19% | 36.19% | 13.81 pp | -29 | 5 | -5.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 103 | 56 | 47 | 54.37% | 54.37% | 54.37% | 4.37 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 8 | -0.62 |
| BTC Market Hours | lstm | LSTM | 103 | 42 | 61 | 40.78% | 40.78% | 40.78% | 9.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | transformer | Transformer | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 103 | 39 | 64 | 37.86% | 37.86% | 37.86% | 12.14 pp | -25 | 8 | -3.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | rf | RandomForest | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 103 | 40 | 63 | 38.83% | 38.83% | 38.83% | 11.17 pp | -23 | 9 | -2.56 |
| BTC Market Hours Daily | xgb | XGBoost | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 9 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
