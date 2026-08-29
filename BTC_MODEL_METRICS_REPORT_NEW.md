# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T03:43:45.314446+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 42 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 138 | 78 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 144 | 66 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 144 | 66 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 12:00:00+00:00 | 51 | 51 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 12:00:00+00:00 | 51 | 51 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 12:00:00+00:00 | 51 | 1 | 50 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 12:00:00+00:00 | 51 | 1 | 50 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 66 | 37 | 29 | 56.06% | 56.06% | 56.06% | 6.06 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| BTC Market Hours Daily | transformer | Transformer | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| BTC Daily | transformer | Transformer | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 68 | 33 | 35 | 48.53% | 48.53% | 48.53% | 1.47 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 68 | 33 | 35 | 48.53% | 48.53% | 48.53% | 1.47 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 6 | -1.17 |
| BTC Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 6 | -1.83 |
| BTC Hourly | lstm | LSTM | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours | lstm | LSTM | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 6 | -3.00 |
| BTC Daily | rf | RandomForest | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 68 | 24 | 44 | 35.29% | 35.29% | 35.29% | 14.71 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 78 | 26 | 52 | 33.33% | 33.33% | 33.33% | 16.67 pp | -26 | 5 | -5.20 |
| BTC Hourly | rf | RandomForest | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 42 | 14 | 28 | 33.33% | 33.33% | 33.33% | 16.67 pp | -14 | 2 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 42 | 14 | 28 | 33.33% | 33.33% | 33.33% | 16.67 pp | -14 | 2 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 68 | 33 | 35 | 48.53% | 48.53% | 48.53% | 1.47 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 68 | 33 | 35 | 48.53% | 48.53% | 48.53% | 1.47 pp | -2 | 4 | -0.50 |
| BTC Daily | rf | RandomForest | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 68 | 24 | 44 | 35.29% | 35.29% | 35.29% | 14.71 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 78 | 26 | 52 | 33.33% | 33.33% | 33.33% | 16.67 pp | -26 | 5 | -5.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 66 | 37 | 29 | 56.06% | 56.06% | 56.06% | 6.06 pp | 8 | 6 | 1.33 |
| BTC Market Hours | rf | RandomForest | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 6 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | lstm | LSTM | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 6 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
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
