# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T10:03:04.857559+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 09:00:00+00:00 | 96 | 1 | 95 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 09:00:00+00:00 | 88 | 34 | 54 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 56 | 22 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 56 | 22 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 14 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 14 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 0 | 14 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 0 | 14 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 22 | 14 | 8 | 63.64% | 63.64% | 63.64% | 13.64 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Market Hours | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 2 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | nn | NN | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| BTC Daily | rf | RandomForest | 24 | 9 | 15 | 37.50% | 37.50% | 37.50% | 12.50 pp | -6 | 2 | -3.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| Consolidated Hourly | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| BTC Market Hours | lstm | LSTM | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 2 | 1.00 |
| BTC Daily | transformer | Transformer | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Daily | rf | RandomForest | 24 | 9 | 15 | 37.50% | 37.50% | 37.50% | 12.50 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 22 | 14 | 8 | 63.64% | 63.64% | 63.64% | 13.64 pp | 6 | 2 | 3.00 |
| BTC Market Hours | nn | NN | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Market Hours | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| BTC Market Hours | lstm | LSTM | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | nn | NN | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | lstm | LSTM | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

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
