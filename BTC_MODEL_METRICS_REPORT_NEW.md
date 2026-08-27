# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T04:31:39.440086+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 6 | 96 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 42 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 82 | 30 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 82 | 30 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 15:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 15:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 15:00:00+00:00 | 21 | 1 | 20 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 15:00:00+00:00 | 21 | 1 | 20 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 32 | 19 | 13 | 59.38% | 59.38% | 59.38% | 9.38 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| BTC Market Hours | nn | NN | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 3 | 1.33 |
| BTC Market Hours | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 2 | 1.00 |
| BTC Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | rf | RandomForest | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Daily | nn | NN | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | nn | NN | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Daily | rf | RandomForest | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Hourly | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |
| Consolidated Hourly | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 3 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 3 | -3.33 |
| BTC Market Hours | lstm | LSTM | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 32 | 12 | 20 | 37.50% | 37.50% | 37.50% | 12.50 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 32 | 19 | 13 | 59.38% | 59.38% | 59.38% | 9.38 pp | 6 | 2 | 3.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | rf | RandomForest | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 2 | -2.00 |
| BTC Daily | xgb | XGBoost | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 32 | 12 | 20 | 37.50% | 37.50% | 37.50% | 12.50 pp | -8 | 2 | -4.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 3 | 1.33 |
| BTC Market Hours | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 3 | 1.33 |
| BTC Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | lstm | LSTM | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 3 | -3.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | nn | NN | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | lstm | LSTM | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 3 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 3 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
