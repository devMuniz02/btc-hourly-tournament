# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T13:57:38.729435+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 12:00:00+00:00 | 99 | 1 | 98 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 12:00:00+00:00 | 94 | 37 | 57 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 12:00:00+00:00 | 60 | 25 | 35 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 12:00:00+00:00 | 60 | 25 | 35 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 1 | 16 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 1 | 16 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| BTC Market Hours | xgb | XGBoost | 25 | 15 | 10 | 60.00% | 60.00% | 60.00% | 10.00 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Market Hours | rf | RandomForest | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 3 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | nn | NN | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | transformer | Transformer | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 25 | 10 | 15 | 40.00% | 40.00% | 40.00% | 10.00 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 3 | -3.00 |
| BTC Daily | rf | RandomForest | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |
| BTC Market Hours | lstm | LSTM | 25 | 6 | 19 | 24.00% | 24.00% | 24.00% | 26.00 pp | -13 | 2 | -6.50 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | nn | NN | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| BTC Daily | rf | RandomForest | 27 | 10 | 17 | 37.04% | 37.04% | 37.04% | 12.96 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 25 | 15 | 10 | 60.00% | 60.00% | 60.00% | 10.00 pp | 5 | 2 | 2.50 |
| BTC Market Hours | nn | NN | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Market Hours | rf | RandomForest | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 2 | 1.50 |
| BTC Market Hours | transformer | Transformer | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 25 | 10 | 15 | 40.00% | 40.00% | 40.00% | 10.00 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 25 | 6 | 19 | 24.00% | 24.00% | 24.00% | 26.00 pp | -13 | 2 | -6.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 14 | 11 | 56.00% | 56.00% | 56.00% | 6.00 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | nn | NN | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 3 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
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
