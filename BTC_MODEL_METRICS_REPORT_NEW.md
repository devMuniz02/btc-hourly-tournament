# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T14:12:00.308361+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 129 | 69 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 165 | 105 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 186 | 93 | 93 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 186 | 93 | 93 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 13:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 13:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 13:00:00+00:00 | 74 | 1 | 73 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 13:00:00+00:00 | 74 | 1 | 73 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Hourly | transformer | Transformer | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 8 | 0.25 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Hourly | nn | NN | 69 | 34 | 35 | 49.28% | 49.28% | 49.28% | 0.72 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| BTC Daily | nn | NN | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 93 | 40 | 53 | 43.01% | 43.01% | 43.01% | 6.99 pp | -13 | 9 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| BTC Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 9 | -2.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 9 | -2.56 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Hourly | rf | RandomForest | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Hourly | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |
| BTC Hourly | xgb | XGBoost | 69 | 21 | 48 | 30.43% | 30.43% | 30.43% | 19.57 pp | -27 | 3 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 3 | 1.00 |
| BTC Hourly | nn | NN | 69 | 34 | 35 | 49.28% | 49.28% | 49.28% | 0.72 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 3 | -2.33 |
| BTC Hourly | rf | RandomForest | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 3 | -4.33 |
| BTC Hourly | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 3 | -5.67 |
| BTC Hourly | xgb | XGBoost | 69 | 21 | 48 | 30.43% | 30.43% | 30.43% | 19.57 pp | -27 | 3 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | nn | NN | 93 | 40 | 53 | 43.01% | 43.01% | 43.01% | 6.99 pp | -13 | 9 | -1.44 |
| BTC Market Hours Daily | lstm | LSTM | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 9 | -2.56 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
