# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T05:04:39.389517+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 7 | 95 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 103 | 43 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 83 | 31 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 82 | 30 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 33 | 20 | 13 | 60.61% | 60.61% | 60.61% | 10.61 pp | 7 | 2 | 3.50 |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| BTC Market Hours | rf | RandomForest | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| BTC Market Hours | nn | NN | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 4 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| BTC Daily | nn | NN | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| BTC Market Hours | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | nn | NN | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Hourly | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Daily | rf | RandomForest | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| BTC Market Hours | lstm | LSTM | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 2 | -4.50 |
| BTC Hourly | rf | RandomForest | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 33 | 20 | 13 | 60.61% | 60.61% | 60.61% | 10.61 pp | 7 | 2 | 3.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| BTC Daily | nn | NN | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| BTC Daily | rf | RandomForest | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 2 | -4.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 3 | 1.67 |
| BTC Market Hours | nn | NN | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| BTC Market Hours | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | lstm | LSTM | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |

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
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

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
