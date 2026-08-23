# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T19:55:18.067623+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 5 | 97 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 100 | 40 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 18:00:00+00:00 | 69 | 28 | 41 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 18:00:00+00:00 | 69 | 28 | 41 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 02:00:00+00:00 | 18 | 18 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 02:00:00+00:00 | 18 | 18 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 02:00:00+00:00 | 18 | 0 | 18 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 02:00:00+00:00 | 18 | 0 | 18 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| BTC Market Hours | rf | RandomForest | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 3 | 2.00 |
| BTC Daily | transformer | Transformer | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| BTC Market Hours | nn | NN | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 3 | 1.33 |
| BTC Market Hours | xgb | XGBoost | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 2 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| BTC Daily | nn | NN | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 2 | -2.00 |
| Consolidated Hourly | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 3 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 28 | 9 | 19 | 32.14% | 32.14% | 32.14% | 17.86 pp | -10 | 4 | -2.50 |
| BTC Hourly | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | lstm | LSTM | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 2 | -4.00 |
| BTC Daily | xgb | XGBoost | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |
| BTC Market Hours | lstm | LSTM | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 5 | 0 | 5 | 0.00% | 0.00% | 0.00% | 50.00 pp | -5 | 1 | -5.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 2 | 2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 2 | -2.00 |
| BTC Daily | lstm | LSTM | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 2 | -4.00 |
| BTC Daily | xgb | XGBoost | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 3 | -4.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 3 | 2.00 |
| BTC Market Hours | nn | NN | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 3 | 1.33 |
| BTC Market Hours | xgb | XGBoost | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 3 | 1.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | lstm | LSTM | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | nn | NN | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 28 | 9 | 19 | 32.14% | 32.14% | 32.14% | 17.86 pp | -10 | 4 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 3 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 3 | -2.00 |

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
