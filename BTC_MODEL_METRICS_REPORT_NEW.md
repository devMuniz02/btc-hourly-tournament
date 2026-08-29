# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T12:49:10.164836+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 109 | 49 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 145 | 85 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 151 | 73 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 151 | 73 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 08:00:00+00:00 | 57 | 57 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 08:00:00+00:00 | 57 | 57 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 08:00:00+00:00 | 57 | 0 | 57 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 08:00:00+00:00 | 57 | 0 | 57 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 73 | 42 | 31 | 57.53% | 57.53% | 57.53% | 7.53 pp | 11 | 6 | 1.83 |
| Consolidated Hourly | rf | RandomForest | 57 | 33 | 24 | 57.89% | 57.89% | 57.89% | 7.89 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 57 | 33 | 24 | 57.89% | 57.89% | 57.89% | 7.89 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 57 | 30 | 27 | 52.63% | 52.63% | 52.63% | 2.63 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 57 | 30 | 27 | 52.63% | 52.63% | 52.63% | 2.63 pp | 3 | 6 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 75 | 38 | 37 | 50.67% | 50.67% | 50.67% | 0.67 pp | 1 | 4 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | transformer | Transformer | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 7 | 0.14 |
| BTC Hourly | nn | NN | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 73 | 35 | 38 | 47.95% | 47.95% | 47.95% | 2.05 pp | -3 | 6 | -0.50 |
| BTC Market Hours | rf | RandomForest | 73 | 35 | 38 | 47.95% | 47.95% | 47.95% | 2.05 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 6 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 75 | 36 | 39 | 48.00% | 48.00% | 48.00% | 2.00 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 75 | 36 | 39 | 48.00% | 48.00% | 48.00% | 2.00 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | nn | NN | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 7 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 3 | -3.00 |
| BTC Daily | rf | RandomForest | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 75 | 27 | 48 | 36.00% | 36.00% | 36.00% | 14.00 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 85 | 27 | 58 | 31.76% | 31.76% | 31.76% | 18.24 pp | -31 | 5 | -6.20 |
| BTC Hourly | xgb | XGBoost | 49 | 15 | 34 | 30.61% | 30.61% | 30.61% | 19.39 pp | -19 | 3 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 3 | 0.33 |
| BTC Hourly | nn | NN | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Hourly | lstm | LSTM | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 49 | 15 | 34 | 30.61% | 30.61% | 30.61% | 19.39 pp | -19 | 3 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 75 | 38 | 37 | 50.67% | 50.67% | 50.67% | 0.67 pp | 1 | 4 | 0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 75 | 36 | 39 | 48.00% | 48.00% | 48.00% | 2.00 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 75 | 36 | 39 | 48.00% | 48.00% | 48.00% | 2.00 pp | -3 | 4 | -0.75 |
| BTC Daily | rf | RandomForest | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 75 | 27 | 48 | 36.00% | 36.00% | 36.00% | 14.00 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 85 | 27 | 58 | 31.76% | 31.76% | 31.76% | 18.24 pp | -31 | 5 | -6.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 73 | 42 | 31 | 57.53% | 57.53% | 57.53% | 7.53 pp | 11 | 6 | 1.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 73 | 35 | 38 | 47.95% | 47.95% | 47.95% | 2.05 pp | -3 | 6 | -0.50 |
| BTC Market Hours | rf | RandomForest | 73 | 35 | 38 | 47.95% | 47.95% | 47.95% | 2.05 pp | -3 | 6 | -0.50 |
| BTC Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | transformer | Transformer | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | nn | NN | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 7 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 57 | 33 | 24 | 57.89% | 57.89% | 57.89% | 7.89 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 57 | 30 | 27 | 52.63% | 52.63% | 52.63% | 2.63 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 57 | 33 | 24 | 57.89% | 57.89% | 57.89% | 7.89 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 57 | 30 | 27 | 52.63% | 52.63% | 52.63% | 2.63 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 6 | -2.83 |

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
