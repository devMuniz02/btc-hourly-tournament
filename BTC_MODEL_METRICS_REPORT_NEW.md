# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T16:14:22.038246+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 112 | 52 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 148 | 88 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 158 | 76 | 82 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 158 | 76 | 82 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 1 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 1 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 76 | 45 | 31 | 59.21% | 59.21% | 59.21% | 9.21 pp | 14 | 6 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | rf | RandomForest | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 7 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 4 | -0.50 |
| BTC Daily | transformer | Transformer | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 78 | 37 | 41 | 47.44% | 47.44% | 47.44% | 2.56 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 7 | -1.14 |
| BTC Hourly | nn | NN | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | nn | NN | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 7 | -2.29 |
| BTC Market Hours | transformer | Transformer | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 78 | 27 | 51 | 34.62% | 34.62% | 34.62% | 15.38 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 52 | 16 | 36 | 30.77% | 30.77% | 30.77% | 19.23 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 88 | 27 | 61 | 30.68% | 30.68% | 30.68% | 19.32 pp | -34 | 5 | -6.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 3 | -1.33 |
| BTC Hourly | lstm | LSTM | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | xgb | XGBoost | 52 | 16 | 36 | 30.77% | 30.77% | 30.77% | 19.23 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 4 | -0.50 |
| BTC Daily | transformer | Transformer | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 78 | 37 | 41 | 47.44% | 47.44% | 47.44% | 2.56 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 78 | 27 | 51 | 34.62% | 34.62% | 34.62% | 15.38 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 88 | 27 | 61 | 30.68% | 30.68% | 30.68% | 19.32 pp | -34 | 5 | -6.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 76 | 45 | 31 | 59.21% | 59.21% | 59.21% | 9.21 pp | 14 | 6 | 2.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | rf | RandomForest | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | lstm | LSTM | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | rf | RandomForest | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 7 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
