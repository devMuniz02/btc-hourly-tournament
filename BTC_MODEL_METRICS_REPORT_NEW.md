# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T15:15:50.353837+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 147 | 87 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 183 | 123 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 14:00:00+00:00 | 218 | 111 | 107 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 14:00:00+00:00 | 218 | 111 | 107 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 90 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 111 | 59 | 52 | 53.15% | 53.15% | 53.15% | 3.15 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 111 | 57 | 54 | 51.35% | 51.35% | 51.35% | 1.35 pp | 3 | 10 | 0.30 |
| BTC Hourly | transformer | Transformer | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 4 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 113 | 56 | 57 | 49.56% | 49.56% | 49.56% | 0.44 pp | -1 | 6 | -0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| BTC Hourly | nn | NN | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 111 | 47 | 64 | 42.34% | 42.34% | 42.34% | 7.66 pp | -17 | 9 | -1.89 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 111 | 45 | 66 | 40.54% | 40.54% | 40.54% | 9.46 pp | -21 | 9 | -2.33 |
| BTC Daily | transformer | Transformer | 113 | 49 | 64 | 43.36% | 43.36% | 43.36% | 6.64 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 111 | 43 | 68 | 38.74% | 38.74% | 38.74% | 11.26 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 111 | 44 | 67 | 39.64% | 39.64% | 39.64% | 10.36 pp | -23 | 9 | -2.56 |
| BTC Market Hours Daily | lstm | LSTM | 111 | 42 | 69 | 37.84% | 37.84% | 37.84% | 12.16 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 113 | 46 | 67 | 40.71% | 40.71% | 40.71% | 9.29 pp | -21 | 6 | -3.50 |
| BTC Hourly | rf | RandomForest | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 123 | 45 | 78 | 36.59% | 36.59% | 36.59% | 13.41 pp | -33 | 7 | -4.71 |
| BTC Daily | lstm | LSTM | 113 | 40 | 73 | 35.40% | 35.40% | 35.40% | 14.60 pp | -33 | 6 | -5.50 |
| BTC Hourly | xgb | XGBoost | 87 | 29 | 58 | 33.33% | 33.33% | 33.33% | 16.67 pp | -29 | 4 | -7.25 |
| BTC Hourly | lstm | LSTM | 87 | 28 | 59 | 32.18% | 32.18% | 32.18% | 17.82 pp | -31 | 4 | -7.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 4 | 0.25 |
| BTC Hourly | nn | NN | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 4 | -4.25 |
| BTC Hourly | xgb | XGBoost | 87 | 29 | 58 | 33.33% | 33.33% | 33.33% | 16.67 pp | -29 | 4 | -7.25 |
| BTC Hourly | lstm | LSTM | 87 | 28 | 59 | 32.18% | 32.18% | 32.18% | 17.82 pp | -31 | 4 | -7.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 113 | 56 | 57 | 49.56% | 49.56% | 49.56% | 0.44 pp | -1 | 6 | -0.17 |
| BTC Daily | nn | NN | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 6 | -0.83 |
| BTC Daily | transformer | Transformer | 113 | 49 | 64 | 43.36% | 43.36% | 43.36% | 6.64 pp | -15 | 6 | -2.50 |
| BTC Daily | rf | RandomForest | 113 | 46 | 67 | 40.71% | 40.71% | 40.71% | 9.29 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 123 | 45 | 78 | 36.59% | 36.59% | 36.59% | 13.41 pp | -33 | 7 | -4.71 |
| BTC Daily | lstm | LSTM | 113 | 40 | 73 | 35.40% | 35.40% | 35.40% | 14.60 pp | -33 | 6 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 111 | 59 | 52 | 53.15% | 53.15% | 53.15% | 3.15 pp | 7 | 9 | 0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 9 | -0.56 |
| BTC Market Hours | transformer | Transformer | 111 | 47 | 64 | 42.34% | 42.34% | 42.34% | 7.66 pp | -17 | 9 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 111 | 45 | 66 | 40.54% | 40.54% | 40.54% | 9.46 pp | -21 | 9 | -2.33 |
| BTC Market Hours | lstm | LSTM | 111 | 44 | 67 | 39.64% | 39.64% | 39.64% | 10.36 pp | -23 | 9 | -2.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 111 | 57 | 54 | 51.35% | 51.35% | 51.35% | 1.35 pp | 3 | 10 | 0.30 |
| BTC Market Hours Daily | transformer | Transformer | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 111 | 43 | 68 | 38.74% | 38.74% | 38.74% | 11.26 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 111 | 42 | 69 | 37.84% | 37.84% | 37.84% | 12.16 pp | -27 | 10 | -2.70 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
