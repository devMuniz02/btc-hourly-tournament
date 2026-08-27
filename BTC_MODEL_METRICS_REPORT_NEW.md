# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T22:33:45.419481+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 20 | 82 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 115 | 55 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 21:00:00+00:00 | 105 | 43 | 62 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 21:00:00+00:00 | 105 | 43 | 62 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 0 | 31 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 0 | 31 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| BTC Market Hours | nn | NN | 43 | 24 | 19 | 55.81% | 55.81% | 55.81% | 5.81 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| BTC Market Hours | rf | RandomForest | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | nn | NN | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 3 | -2.33 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 4 | -3.75 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 43 | 11 | 32 | 25.58% | 25.58% | 25.58% | 24.42 pp | -21 | 5 | -4.20 |
| BTC Market Hours | lstm | LSTM | 43 | 13 | 30 | 30.23% | 30.23% | 30.23% | 19.77 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 3 | -5.00 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 3 | 0.33 |
| BTC Daily | nn | NN | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 3 | -2.33 |
| BTC Daily | xgb | XGBoost | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 3 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 43 | 24 | 19 | 55.81% | 55.81% | 55.81% | 5.81 pp | 5 | 4 | 1.25 |
| BTC Market Hours | rf | RandomForest | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 4 | -0.25 |
| BTC Market Hours | transformer | Transformer | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 43 | 13 | 30 | 30.23% | 30.23% | 30.23% | 19.77 pp | -17 | 4 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | nn | NN | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 43 | 11 | 32 | 25.58% | 25.58% | 25.58% | 24.42 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

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
