# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T00:37:01.440599+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 251 | 191 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 287 | 227 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 409 | 215 | 194 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 409 | 215 | 194 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 191 | 98 | 93 | 51.31% | 51.31% | 51.31% | 1.31 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | transformer | Transformer | 215 | 110 | 105 | 51.16% | 51.16% | 51.16% | 1.16 pp | 5 | 18 | 0.28 |
| BTC Market Hours | nn | NN | 215 | 109 | 106 | 50.70% | 50.70% | 50.70% | 0.70 pp | 3 | 17 | 0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 18 | -0.39 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 17 | -0.41 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 18 | -0.72 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 17 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| BTC Market Hours | rf | RandomForest | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 17 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 217 | 102 | 115 | 47.00% | 47.00% | 47.00% | 3.00 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 18 | -1.39 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 17 | -1.71 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Daily | nn | NN | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 10 | -2.10 |
| BTC Hourly | transformer | Transformer | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 18 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Market Hours | lstm | LSTM | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 17 | -2.29 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 215 | 85 | 130 | 39.53% | 39.53% | 39.53% | 10.47 pp | -45 | 18 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 217 | 89 | 128 | 41.01% | 41.01% | 41.01% | 8.99 pp | -39 | 10 | -3.90 |
| BTC Hourly | rf | RandomForest | 191 | 79 | 112 | 41.36% | 41.36% | 41.36% | 8.64 pp | -33 | 8 | -4.12 |
| BTC Daily | rf | RandomForest | 217 | 84 | 133 | 38.71% | 38.71% | 38.71% | 11.29 pp | -49 | 10 | -4.90 |
| BTC Hourly | lstm | LSTM | 191 | 72 | 119 | 37.70% | 37.70% | 37.70% | 12.30 pp | -47 | 8 | -5.88 |
| BTC Daily | xgb | XGBoost | 227 | 81 | 146 | 35.68% | 35.68% | 35.68% | 14.32 pp | -65 | 11 | -5.91 |
| BTC Hourly | xgb | XGBoost | 191 | 70 | 121 | 36.65% | 36.65% | 36.65% | 13.35 pp | -51 | 8 | -6.38 |
| BTC Daily | lstm | LSTM | 217 | 73 | 144 | 33.64% | 33.64% | 33.64% | 16.36 pp | -71 | 10 | -7.10 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 191 | 98 | 93 | 51.31% | 51.31% | 51.31% | 1.31 pp | 5 | 8 | 0.62 |
| BTC Hourly | transformer | Transformer | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 8 | -2.12 |
| BTC Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 191 | 79 | 112 | 41.36% | 41.36% | 41.36% | 8.64 pp | -33 | 8 | -4.12 |
| BTC Hourly | lstm | LSTM | 191 | 72 | 119 | 37.70% | 37.70% | 37.70% | 12.30 pp | -47 | 8 | -5.88 |
| BTC Hourly | xgb | XGBoost | 191 | 70 | 121 | 36.65% | 36.65% | 36.65% | 13.35 pp | -51 | 8 | -6.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 217 | 102 | 115 | 47.00% | 47.00% | 47.00% | 3.00 pp | -13 | 10 | -1.30 |
| BTC Daily | nn | NN | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 10 | -2.10 |
| BTC Daily | transformer | Transformer | 217 | 89 | 128 | 41.01% | 41.01% | 41.01% | 8.99 pp | -39 | 10 | -3.90 |
| BTC Daily | rf | RandomForest | 217 | 84 | 133 | 38.71% | 38.71% | 38.71% | 11.29 pp | -49 | 10 | -4.90 |
| BTC Daily | xgb | XGBoost | 227 | 81 | 146 | 35.68% | 35.68% | 35.68% | 14.32 pp | -65 | 11 | -5.91 |
| BTC Daily | lstm | LSTM | 217 | 73 | 144 | 33.64% | 33.64% | 33.64% | 16.36 pp | -71 | 10 | -7.10 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 215 | 109 | 106 | 50.70% | 50.70% | 50.70% | 0.70 pp | 3 | 17 | 0.18 |
| BTC Market Hours | transformer | Transformer | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 17 | -0.41 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 17 | -0.88 |
| BTC Market Hours | rf | RandomForest | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 17 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 17 | -1.71 |
| BTC Market Hours | lstm | LSTM | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 17 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 215 | 110 | 105 | 51.16% | 51.16% | 51.16% | 1.16 pp | 5 | 18 | 0.28 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 18 | -0.39 |
| BTC Market Hours Daily | nn | NN | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 18 | -0.72 |
| BTC Market Hours Daily | rf | RandomForest | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 18 | -1.39 |
| BTC Market Hours Daily | xgb | XGBoost | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 18 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 215 | 85 | 130 | 39.53% | 39.53% | 39.53% | 10.47 pp | -45 | 18 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
