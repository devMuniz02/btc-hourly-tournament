# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T23:26:38.005220+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 118 | 58 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 153 | 93 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 22:00:00+00:00 | 170 | 81 | 89 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 22:00:00+00:00 | 170 | 81 | 89 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 0 | 63 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 0 | 63 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 81 | 47 | 34 | 58.02% | 58.02% | 58.02% | 8.02 pp | 13 | 7 | 1.86 |
| Consolidated Hourly | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 42 | 39 | 51.85% | 51.85% | 51.85% | 1.85 pp | 3 | 8 | 0.38 |
| BTC Hourly | nn | NN | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| BTC Daily | transformer | Transformer | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| BTC Daily | nn | NN | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 8 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 81 | 34 | 47 | 41.98% | 41.98% | 41.98% | 8.02 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| BTC Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| BTC Hourly | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 3 | -4.67 |
| BTC Daily | rf | RandomForest | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 4 | -5.75 |
| BTC Hourly | xgb | XGBoost | 58 | 19 | 39 | 32.76% | 32.76% | 32.76% | 17.24 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 93 | 27 | 66 | 29.03% | 29.03% | 29.03% | 20.97 pp | -39 | 5 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 3 | -4.67 |
| BTC Hourly | xgb | XGBoost | 58 | 19 | 39 | 32.76% | 32.76% | 32.76% | 17.24 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 4 | -1.75 |
| BTC Daily | rf | RandomForest | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 4 | -5.75 |
| BTC Daily | xgb | XGBoost | 93 | 27 | 66 | 29.03% | 29.03% | 29.03% | 20.97 pp | -39 | 5 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 81 | 47 | 34 | 58.02% | 58.02% | 58.02% | 8.02 pp | 13 | 7 | 1.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 7 | -1.29 |
| BTC Market Hours | transformer | Transformer | 81 | 34 | 47 | 41.98% | 41.98% | 41.98% | 8.02 pp | -13 | 7 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 42 | 39 | 51.85% | 51.85% | 51.85% | 1.85 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 8 | -0.12 |
| BTC Market Hours Daily | rf | RandomForest | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |

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
