# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T01:03:51.875747+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 119 | 59 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 155 | 95 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 174 | 83 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 174 | 83 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| BTC Market Hours | nn | NN | 83 | 47 | 36 | 56.63% | 56.63% | 56.63% | 6.63 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 7 | -0.14 |
| BTC Hourly | nn | NN | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Market Hours | rf | RandomForest | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 7 | -1.57 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 3 | -1.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| BTC Hourly | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 4 | -4.75 |
| BTC Hourly | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 3 | -5.00 |
| BTC Daily | lstm | LSTM | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 59 | 19 | 40 | 32.20% | 32.20% | 32.20% | 17.80 pp | -21 | 3 | -7.00 |
| BTC Daily | xgb | XGBoost | 95 | 29 | 66 | 30.53% | 30.53% | 30.53% | 19.47 pp | -37 | 5 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 3 | 0.33 |
| BTC Hourly | nn | NN | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 3 | -1.67 |
| BTC Hourly | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 3 | -5.00 |
| BTC Hourly | xgb | XGBoost | 59 | 19 | 40 | 32.20% | 32.20% | 32.20% | 17.80 pp | -21 | 3 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 4 | -1.75 |
| BTC Daily | rf | RandomForest | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 4 | -6.25 |
| BTC Daily | xgb | XGBoost | 95 | 29 | 66 | 30.53% | 30.53% | 30.53% | 19.47 pp | -37 | 5 | -7.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 83 | 47 | 36 | 56.63% | 56.63% | 56.63% | 6.63 pp | 11 | 7 | 1.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 7 | -0.43 |
| BTC Market Hours | lstm | LSTM | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 8 | -0.12 |
| BTC Market Hours Daily | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 83 | 36 | 47 | 43.37% | 43.37% | 43.37% | 6.63 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 8 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

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
