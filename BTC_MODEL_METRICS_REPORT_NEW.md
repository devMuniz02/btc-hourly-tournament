# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T07:16:43.660269+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 06:00:00+00:00 | 93 | 1 | 92 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 06:00:00+00:00 | 84 | 33 | 51 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 55 | 21 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 55 | 21 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 2 | 3.50 |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| BTC Market Hours | rf | RandomForest | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 2 | 1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| BTC Market Hours | nn | NN | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 23 | 9 | 14 | 39.13% | 39.13% | 39.13% | 10.87 pp | -5 | 2 | -2.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| BTC Daily | lstm | LSTM | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| BTC Market Hours | lstm | LSTM | 21 | 5 | 16 | 23.81% | 23.81% | 23.81% | 26.19 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 2 | 1.50 |
| BTC Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Daily | rf | RandomForest | 23 | 9 | 14 | 39.13% | 39.13% | 39.13% | 10.87 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Daily | lstm | LSTM | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 2 | 3.50 |
| BTC Market Hours | rf | RandomForest | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 21 | 5 | 16 | 23.81% | 23.81% | 23.81% | 26.19 pp | -11 | 2 | -5.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | xgb | XGBoost | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | nn | NN | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
