# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T19:50:37.796218+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 36 | 66 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 132 | 72 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 18:00:00+00:00 | 132 | 60 | 72 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 18:00:00+00:00 | 132 | 60 | 72 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 19:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 19:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 19:00:00+00:00 | 47 | 1 | 46 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 19:00:00+00:00 | 47 | 1 | 46 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 60 | 33 | 27 | 55.00% | 55.00% | 55.00% | 5.00 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| BTC Hourly | nn | NN | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 2 | 1.00 |
| BTC Daily | transformer | Transformer | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 31 | 29 | 51.67% | 51.67% | 51.67% | 1.67 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 60 | 31 | 29 | 51.67% | 51.67% | 51.67% | 1.67 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 60 | 28 | 32 | 46.67% | 46.67% | 46.67% | 3.33 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Daily | nn | NN | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 3 | -2.00 |
| BTC Market Hours | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 6 | -2.00 |
| BTC Hourly | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 2 | -3.00 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| BTC Market Hours | lstm | LSTM | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 5 | -3.20 |
| BTC Market Hours Daily | lstm | LSTM | 60 | 20 | 40 | 33.33% | 33.33% | 33.33% | 16.67 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 62 | 23 | 39 | 37.10% | 37.10% | 37.10% | 12.90 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 72 | 24 | 48 | 33.33% | 33.33% | 33.33% | 16.67 pp | -24 | 4 | -6.00 |
| BTC Hourly | rf | RandomForest | 36 | 10 | 26 | 27.78% | 27.78% | 27.78% | 22.22 pp | -16 | 2 | -8.00 |
| BTC Hourly | xgb | XGBoost | 36 | 10 | 26 | 27.78% | 27.78% | 27.78% | 22.22 pp | -16 | 2 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 2 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 2 | -3.00 |
| BTC Hourly | rf | RandomForest | 36 | 10 | 26 | 27.78% | 27.78% | 27.78% | 22.22 pp | -16 | 2 | -8.00 |
| BTC Hourly | xgb | XGBoost | 36 | 10 | 26 | 27.78% | 27.78% | 27.78% | 22.22 pp | -16 | 2 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 3 | -2.00 |
| BTC Daily | rf | RandomForest | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 62 | 23 | 39 | 37.10% | 37.10% | 37.10% | 12.90 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 72 | 24 | 48 | 33.33% | 33.33% | 33.33% | 16.67 pp | -24 | 4 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 60 | 33 | 27 | 55.00% | 55.00% | 55.00% | 5.00 pp | 6 | 5 | 1.20 |
| BTC Market Hours | rf | RandomForest | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 60 | 28 | 32 | 46.67% | 46.67% | 46.67% | 3.33 pp | -4 | 5 | -0.80 |
| BTC Market Hours | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 5 | -3.20 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 31 | 29 | 51.67% | 51.67% | 51.67% | 1.67 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 60 | 31 | 29 | 51.67% | 51.67% | 51.67% | 1.67 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 6 | -1.00 |
| BTC Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 60 | 20 | 40 | 33.33% | 33.33% | 33.33% | 16.67 pp | -20 | 6 | -3.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
