# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T05:28:04.952768+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 141 | 81 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 177 | 117 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 209 | 105 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 208 | 104 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| BTC Market Hours | nn | NN | 105 | 57 | 48 | 54.29% | 54.29% | 54.29% | 4.29 pp | 9 | 9 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 5 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 81 | 42 | 39 | 51.85% | 51.85% | 51.85% | 1.85 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| BTC Daily | nn | NN | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 5 | 0.20 |
| BTC Hourly | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 104 | 46 | 58 | 44.23% | 44.23% | 44.23% | 5.77 pp | -12 | 9 | -1.33 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| BTC Hourly | rf | RandomForest | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 4 | -1.75 |
| BTC Daily | rf | RandomForest | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 5 | -2.20 |
| BTC Daily | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 5 | -2.20 |
| BTC Market Hours | transformer | Transformer | 105 | 42 | 63 | 40.00% | 40.00% | 40.00% | 10.00 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 105 | 42 | 63 | 40.00% | 40.00% | 40.00% | 10.00 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 9 | -3.89 |
| BTC Market Hours Daily | lstm | LSTM | 104 | 33 | 71 | 31.73% | 31.73% | 31.73% | 18.27 pp | -38 | 9 | -4.22 |
| BTC Daily | lstm | LSTM | 107 | 42 | 65 | 39.25% | 39.25% | 39.25% | 10.75 pp | -23 | 5 | -4.60 |
| BTC Daily | xgb | XGBoost | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 6 | -5.17 |
| BTC Hourly | xgb | XGBoost | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 4 | -5.25 |
| BTC Hourly | lstm | LSTM | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 4 | -5.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 4 | 1.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 81 | 42 | 39 | 51.85% | 51.85% | 51.85% | 1.85 pp | 3 | 4 | 0.75 |
| BTC Hourly | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 4 | -0.25 |
| BTC Hourly | rf | RandomForest | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 4 | -1.75 |
| BTC Hourly | xgb | XGBoost | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 4 | -5.25 |
| BTC Hourly | lstm | LSTM | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 4 | -5.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 5 | 1.00 |
| BTC Daily | nn | NN | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 5 | 0.20 |
| BTC Daily | rf | RandomForest | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 5 | -2.20 |
| BTC Daily | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 5 | -2.20 |
| BTC Daily | lstm | LSTM | 107 | 42 | 65 | 39.25% | 39.25% | 39.25% | 10.75 pp | -23 | 5 | -4.60 |
| BTC Daily | xgb | XGBoost | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 6 | -5.17 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 105 | 57 | 48 | 54.29% | 54.29% | 54.29% | 4.29 pp | 9 | 9 | 1.00 |
| BTC Market Hours | rf | RandomForest | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| BTC Market Hours | transformer | Transformer | 105 | 42 | 63 | 40.00% | 40.00% | 40.00% | 10.00 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 105 | 42 | 63 | 40.00% | 40.00% | 40.00% | 10.00 pp | -21 | 9 | -2.33 |
| BTC Market Hours | lstm | LSTM | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 9 | -3.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 9 | -0.67 |
| BTC Market Hours Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 104 | 46 | 58 | 44.23% | 44.23% | 44.23% | 5.77 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | lstm | LSTM | 104 | 33 | 71 | 31.73% | 31.73% | 31.73% | 18.27 pp | -38 | 9 | -4.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
