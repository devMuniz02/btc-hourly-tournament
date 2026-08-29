# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T21:27:53.702667+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 116 | 56 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 152 | 92 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 20:00:00+00:00 | 167 | 80 | 87 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 20:00:00+00:00 | 167 | 80 | 87 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 1 | 62 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 1 | 62 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 80 | 46 | 34 | 57.50% | 57.50% | 57.50% | 7.50 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | transformer | Transformer | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Daily | transformer | Transformer | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Daily | nn | NN | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| BTC Market Hours | lstm | LSTM | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 82 | 32 | 50 | 39.02% | 39.02% | 39.02% | 10.98 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 56 | 18 | 38 | 32.14% | 32.14% | 32.14% | 17.86 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 92 | 28 | 64 | 30.43% | 30.43% | 30.43% | 19.57 pp | -36 | 5 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Hourly | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 3 | -4.00 |
| BTC Hourly | xgb | XGBoost | 56 | 18 | 38 | 32.14% | 32.14% | 32.14% | 17.86 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 82 | 40 | 42 | 48.78% | 48.78% | 48.78% | 1.22 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Daily | nn | NN | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 82 | 32 | 50 | 39.02% | 39.02% | 39.02% | 10.98 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 92 | 28 | 64 | 30.43% | 30.43% | 30.43% | 19.57 pp | -36 | 5 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 80 | 46 | 34 | 57.50% | 57.50% | 57.50% | 7.50 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 7 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
