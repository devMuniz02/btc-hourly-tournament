# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T22:38:22.089495+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 117 | 57 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 153 | 93 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 169 | 81 | 88 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 169 | 81 | 88 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T03:00:00+00:00 | 63 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T03:00:00+00:00 | 63 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T03:00:00+00:00 | 63 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T03:00:00+00:00 | 64 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 81 | 47 | 34 | 58.02% | 58.02% | 58.02% | 8.02 pp | 13 | 7 | 1.86 |
| Consolidated Hourly | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 42 | 39 | 51.85% | 51.85% | 51.85% | 1.85 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 57 | 29 | 28 | 50.88% | 50.88% | 50.88% | 0.88 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| BTC Hourly | nn | NN | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | nn | NN | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | transformer | Transformer | 81 | 34 | 47 | 41.98% | 41.98% | 41.98% | 8.02 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| BTC Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| BTC Hourly | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 3 | -4.33 |
| BTC Daily | rf | RandomForest | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 57 | 19 | 38 | 33.33% | 33.33% | 33.33% | 16.67 pp | -19 | 3 | -6.33 |
| BTC Daily | xgb | XGBoost | 93 | 28 | 65 | 30.11% | 30.11% | 30.11% | 19.89 pp | -37 | 5 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 57 | 29 | 28 | 50.88% | 50.88% | 50.88% | 0.88 pp | 1 | 3 | 0.33 |
| BTC Hourly | nn | NN | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| BTC Hourly | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 3 | -4.33 |
| BTC Hourly | xgb | XGBoost | 57 | 19 | 38 | 33.33% | 33.33% | 33.33% | 16.67 pp | -19 | 3 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 83 | 40 | 43 | 48.19% | 48.19% | 48.19% | 1.81 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 83 | 32 | 51 | 38.55% | 38.55% | 38.55% | 11.45 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 4 | -6.25 |
| BTC Daily | xgb | XGBoost | 93 | 28 | 65 | 30.11% | 30.11% | 30.11% | 19.89 pp | -37 | 5 | -7.40 |

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

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
