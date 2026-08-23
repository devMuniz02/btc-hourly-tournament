# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T03:50:01.002674+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 02:00:00+00:00 | 89 | 1 | 88 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 02:00:00+00:00 | 78 | 31 | 47 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 53 | 19 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 52 | 18 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 05:00:00+00:00 | 10 | 10 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 05:00:00+00:00 | 10 | 10 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 05:00:00+00:00 | 10 | 0 | 10 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 05:00:00+00:00 | 10 | 0 | 10 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 19 | 13 | 6 | 68.42% | 68.42% | 68.42% | 18.42 pp | 7 | 2 | 3.50 |
| BTC Market Hours | rf | RandomForest | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | lstm | LSTM | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 2 | 2.00 |
| BTC Daily | nn | NN | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 2 | 1.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | rf | RandomForest | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | nn | NN | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | xgb | XGBoost | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 3 | -1.33 |
| BTC Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 2 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 2 | -2.00 |
| BTC Daily | xgb | XGBoost | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| BTC Daily | lstm | LSTM | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| BTC Market Hours | lstm | LSTM | 19 | 5 | 14 | 26.32% | 26.32% | 26.32% | 23.68 pp | -9 | 2 | -4.50 |

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
| BTC Daily | nn | NN | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 2 | 1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Daily | xgb | XGBoost | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| BTC Daily | lstm | LSTM | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 19 | 13 | 6 | 68.42% | 68.42% | 68.42% | 18.42 pp | 7 | 2 | 3.50 |
| BTC Market Hours | rf | RandomForest | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 2 | 2.50 |
| BTC Market Hours | nn | NN | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 19 | 5 | 14 | 26.32% | 26.32% | 26.32% | 23.68 pp | -9 | 2 | -4.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | rf | RandomForest | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | nn | NN | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | xgb | XGBoost | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 3 | -1.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 2 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 2 | -2.00 |

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
