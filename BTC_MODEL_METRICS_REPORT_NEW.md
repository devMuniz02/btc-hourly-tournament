# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T04:33:21.588117+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 24 | 78 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 120 | 60 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 113 | 48 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 113 | 48 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 1 | 36 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 1 | 36 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | lstm | LSTM | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 5 | -0.40 |
| BTC Daily | nn | NN | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | transformer | Transformer | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| BTC Daily | rf | RandomForest | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | nn | NN | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |
| BTC Market Hours | xgb | XGBoost | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Hourly | transformer | Transformer | 24 | 10 | 14 | 41.67% | 41.67% | 41.67% | 8.33 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 4 | -4.00 |
| BTC Market Hours | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 48 | 13 | 35 | 27.08% | 27.08% | 27.08% | 22.92 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 3 | -5.33 |
| BTC Hourly | rf | RandomForest | 24 | 9 | 15 | 37.50% | 37.50% | 37.50% | 12.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 1 | 2.00 |
| BTC Hourly | lstm | LSTM | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | transformer | Transformer | 24 | 10 | 14 | 41.67% | 41.67% | 41.67% | 8.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 24 | 9 | 15 | 37.50% | 37.50% | 37.50% | 12.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 60 | 22 | 38 | 36.67% | 36.67% | 36.67% | 13.33 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 3 | -5.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 4 | 1.50 |
| BTC Market Hours | rf | RandomForest | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | transformer | Transformer | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Market Hours | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | nn | NN | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 48 | 13 | 35 | 27.08% | 27.08% | 27.08% | 22.92 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
