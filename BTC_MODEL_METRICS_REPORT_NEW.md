# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T04:21:56.049868+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 204 | 144 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 240 | 180 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 324 | 168 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 324 | 168 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 144 | 76 | 68 | 52.78% | 52.78% | 52.78% | 2.78 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 168 | 87 | 81 | 51.79% | 51.79% | 51.79% | 1.79 pp | 6 | 13 | 0.46 |
| BTC Market Hours Daily | transformer | Transformer | 168 | 82 | 86 | 48.81% | 48.81% | 48.81% | 1.19 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 144 | 71 | 73 | 49.31% | 49.31% | 49.31% | 0.69 pp | -2 | 6 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 168 | 80 | 88 | 47.62% | 47.62% | 47.62% | 2.38 pp | -8 | 14 | -0.57 |
| BTC Market Hours | rf | RandomForest | 168 | 79 | 89 | 47.02% | 47.02% | 47.02% | 2.98 pp | -10 | 13 | -0.77 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | nn | NN | 168 | 78 | 90 | 46.43% | 46.43% | 46.43% | 3.57 pp | -12 | 14 | -0.86 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 14 | -1.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 8 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | xgb | XGBoost | 168 | 71 | 97 | 42.26% | 42.26% | 42.26% | 7.74 pp | -26 | 14 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 8 | -2.25 |
| BTC Market Hours | lstm | LSTM | 168 | 69 | 99 | 41.07% | 41.07% | 41.07% | 8.93 pp | -30 | 13 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | transformer | Transformer | 170 | 75 | 95 | 44.12% | 44.12% | 44.12% | 5.88 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 168 | 66 | 102 | 39.29% | 39.29% | 39.29% | 10.71 pp | -36 | 14 | -2.57 |
| BTC Hourly | nn | NN | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Daily | rf | RandomForest | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 8 | -3.75 |
| BTC Hourly | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 6 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 170 | 60 | 110 | 35.29% | 35.29% | 35.29% | 14.71 pp | -50 | 8 | -6.25 |
| BTC Hourly | lstm | LSTM | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |
| BTC Hourly | xgb | XGBoost | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 144 | 76 | 68 | 52.78% | 52.78% | 52.78% | 2.78 pp | 8 | 6 | 1.33 |
| BTC Hourly | transformer | Transformer | 144 | 71 | 73 | 49.31% | 49.31% | 49.31% | 0.69 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 6 | -3.00 |
| BTC Hourly | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 6 | -4.00 |
| BTC Hourly | lstm | LSTM | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |
| BTC Hourly | xgb | XGBoost | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 170 | 75 | 95 | 44.12% | 44.12% | 44.12% | 5.88 pp | -20 | 8 | -2.50 |
| BTC Daily | rf | RandomForest | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 8 | -3.75 |
| BTC Daily | xgb | XGBoost | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 170 | 60 | 110 | 35.29% | 35.29% | 35.29% | 14.71 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 168 | 87 | 81 | 51.79% | 51.79% | 51.79% | 1.79 pp | 6 | 13 | 0.46 |
| BTC Market Hours | rf | RandomForest | 168 | 79 | 89 | 47.02% | 47.02% | 47.02% | 2.98 pp | -10 | 13 | -0.77 |
| BTC Market Hours | transformer | Transformer | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 13 | -1.85 |
| BTC Market Hours | lstm | LSTM | 168 | 69 | 99 | 41.07% | 41.07% | 41.07% | 8.93 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 168 | 82 | 86 | 48.81% | 48.81% | 48.81% | 1.19 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 168 | 80 | 88 | 47.62% | 47.62% | 47.62% | 2.38 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 168 | 78 | 90 | 46.43% | 46.43% | 46.43% | 3.57 pp | -12 | 14 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 14 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 168 | 71 | 97 | 42.26% | 42.26% | 42.26% | 7.74 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 168 | 66 | 102 | 39.29% | 39.29% | 39.29% | 10.71 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
