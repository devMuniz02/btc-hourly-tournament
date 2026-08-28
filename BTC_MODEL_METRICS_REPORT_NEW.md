# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T23:52:24.953349+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 39 | 63 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 135 | 75 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 22:00:00+00:00 | 139 | 63 | 76 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 22:00:00+00:00 | 139 | 63 | 76 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 22:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 22:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 22:00:00+00:00 | 50 | 1 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 22:00:00+00:00 | 50 | 1 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 63 | 35 | 28 | 55.56% | 55.56% | 55.56% | 5.56 pp | 7 | 5 | 1.40 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Hourly | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 5 | 0.40 |
| BTC Daily | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 4 | 0.25 |
| BTC Market Hours | rf | RandomForest | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Hourly | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 63 | 24 | 39 | 38.10% | 38.10% | 38.10% | 11.90 pp | -15 | 5 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 6 | -3.17 |
| Consolidated Hourly | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |
| BTC Daily | rf | RandomForest | 65 | 25 | 40 | 38.46% | 38.46% | 38.46% | 11.54 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 75 | 25 | 50 | 33.33% | 33.33% | 33.33% | 16.67 pp | -25 | 5 | -5.00 |
| BTC Hourly | rf | RandomForest | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 2 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 2 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 4 | 0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 4 | -0.75 |
| BTC Daily | rf | RandomForest | 65 | 25 | 40 | 38.46% | 38.46% | 38.46% | 11.54 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 75 | 25 | 50 | 33.33% | 33.33% | 33.33% | 16.67 pp | -25 | 5 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 63 | 35 | 28 | 55.56% | 55.56% | 55.56% | 5.56 pp | 7 | 5 | 1.40 |
| BTC Market Hours | rf | RandomForest | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 63 | 24 | 39 | 38.10% | 38.10% | 38.10% | 11.90 pp | -15 | 5 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 6 | -3.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
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
