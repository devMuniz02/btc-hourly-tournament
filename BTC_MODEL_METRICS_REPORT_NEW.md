# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T06:08:33.394689+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 26 | 76 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 121 | 61 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 114 | 49 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 114 | 49 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| BTC Market Hours | nn | NN | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 3 | 0.33 |
| BTC Hourly | lstm | LSTM | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| BTC Market Hours | rf | RandomForest | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| BTC Market Hours Daily | nn | NN | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Hourly | transformer | Transformer | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| BTC Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 26 | 10 | 16 | 38.46% | 38.46% | 38.46% | 11.54 pp | -6 | 2 | -3.00 |
| BTC Hourly | xgb | XGBoost | 26 | 10 | 16 | 38.46% | 38.46% | 38.46% | 11.54 pp | -6 | 2 | -3.00 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |
| BTC Market Hours Daily | lstm | LSTM | 49 | 14 | 35 | 28.57% | 28.57% | 28.57% | 21.43 pp | -21 | 5 | -4.20 |
| BTC Market Hours | lstm | LSTM | 49 | 16 | 33 | 32.65% | 32.65% | 32.65% | 17.35 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 3 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| BTC Hourly | lstm | LSTM | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 26 | 10 | 16 | 38.46% | 38.46% | 38.46% | 11.54 pp | -6 | 2 | -3.00 |
| BTC Hourly | xgb | XGBoost | 26 | 10 | 16 | 38.46% | 38.46% | 38.46% | 11.54 pp | -6 | 2 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 3 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 3 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 4 | 1.25 |
| BTC Market Hours | rf | RandomForest | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| BTC Market Hours | lstm | LSTM | 49 | 16 | 33 | 32.65% | 32.65% | 32.65% | 17.35 pp | -17 | 4 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | rf | RandomForest | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 49 | 14 | 35 | 28.57% | 28.57% | 28.57% | 21.43 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

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
