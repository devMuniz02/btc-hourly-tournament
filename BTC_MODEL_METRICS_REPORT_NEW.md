# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T09:21:55.127900+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 28 | 74 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 124 | 64 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 117 | 52 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 117 | 52 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 00:00:00+00:00 | 39 | 39 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 00:00:00+00:00 | 39 | 39 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 00:00:00+00:00 | 39 | 1 | 38 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 00:00:00+00:00 | 39 | 1 | 38 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 2 | 3.00 |
| BTC Market Hours | nn | NN | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 39 | 23 | 16 | 58.97% | 58.97% | 58.97% | 8.97 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 23 | 16 | 58.97% | 58.97% | 58.97% | 8.97 pp | 7 | 5 | 1.40 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 5 | 0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | nn | NN | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 5 | -0.40 |
| BTC Market Hours | rf | RandomForest | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Hourly | lstm | LSTM | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | nn | NN | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 3 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| BTC Hourly | rf | RandomForest | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Market Hours | lstm | LSTM | 52 | 18 | 34 | 34.62% | 34.62% | 34.62% | 15.38 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 52 | 15 | 37 | 28.85% | 28.85% | 28.85% | 21.15 pp | -22 | 5 | -4.40 |
| BTC Daily | xgb | XGBoost | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 54 | 18 | 36 | 33.33% | 33.33% | 33.33% | 16.67 pp | -18 | 3 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 2 | 3.00 |
| BTC Hourly | nn | NN | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 2 | -1.00 |
| BTC Hourly | rf | RandomForest | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 2 | -4.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 54 | 18 | 36 | 33.33% | 33.33% | 33.33% | 16.67 pp | -18 | 3 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 4 | 1.50 |
| BTC Market Hours | rf | RandomForest | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| BTC Market Hours | lstm | LSTM | 52 | 18 | 34 | 34.62% | 34.62% | 34.62% | 15.38 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | nn | NN | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 52 | 15 | 37 | 28.85% | 28.85% | 28.85% | 21.15 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 39 | 23 | 16 | 58.97% | 58.97% | 58.97% | 8.97 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 23 | 16 | 58.97% | 58.97% | 58.97% | 8.97 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
