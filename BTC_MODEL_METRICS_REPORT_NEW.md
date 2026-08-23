# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T10:29:48.694596+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 09:00:00+00:00 | 89 | 35 | 54 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 57 | 23 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 57 | 23 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 14 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 14 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 0 | 14 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 09:00:00+00:00 | 14 | 0 | 14 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 23 | 14 | 9 | 60.87% | 60.87% | 60.87% | 10.87 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| BTC Market Hours | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| BTC Daily | rf | RandomForest | 25 | 10 | 15 | 40.00% | 40.00% | 40.00% | 10.00 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| Consolidated Hourly | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| BTC Market Hours | lstm | LSTM | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Daily | rf | RandomForest | 25 | 10 | 15 | 40.00% | 40.00% | 40.00% | 10.00 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 23 | 14 | 9 | 60.87% | 60.87% | 60.87% | 10.87 pp | 5 | 2 | 2.50 |
| BTC Market Hours | nn | NN | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| BTC Market Hours | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| BTC Market Hours | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| BTC Market Hours | lstm | LSTM | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | nn | NN | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |

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
