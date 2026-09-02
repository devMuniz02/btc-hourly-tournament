# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T07:51:37.098330+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 175 | 115 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 211 | 151 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 269 | 139 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 269 | 139 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 115 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 115 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 18 | 97 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 18 | 97 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 139 | 72 | 67 | 51.80% | 51.80% | 51.80% | 1.80 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 115 | 57 | 58 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 139 | 67 | 72 | 48.20% | 48.20% | 48.20% | 1.80 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 141 | 68 | 73 | 48.23% | 48.23% | 48.23% | 1.77 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 12 | -0.75 |
| BTC Market Hours | rf | RandomForest | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| BTC Market Hours | transformer | Transformer | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| BTC Hourly | transformer | Transformer | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 5 | -1.40 |
| BTC Hourly | nn | NN | 115 | 53 | 62 | 46.09% | 46.09% | 46.09% | 3.91 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 139 | 57 | 82 | 41.01% | 41.01% | 41.01% | 8.99 pp | -25 | 12 | -2.08 |
| BTC Market Hours | xgb | XGBoost | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 139 | 53 | 86 | 38.13% | 38.13% | 38.13% | 11.87 pp | -33 | 12 | -2.75 |
| BTC Market Hours | lstm | LSTM | 139 | 54 | 85 | 38.85% | 38.85% | 38.85% | 11.15 pp | -31 | 11 | -2.82 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| BTC Hourly | rf | RandomForest | 115 | 49 | 66 | 42.61% | 42.61% | 42.61% | 7.39 pp | -17 | 5 | -3.40 |
| BTC Daily | rf | RandomForest | 141 | 58 | 83 | 41.13% | 41.13% | 41.13% | 8.87 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |
| BTC Daily | xgb | XGBoost | 151 | 54 | 97 | 35.76% | 35.76% | 35.76% | 14.24 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 141 | 49 | 92 | 34.75% | 34.75% | 34.75% | 15.25 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 115 | 42 | 73 | 36.52% | 36.52% | 36.52% | 13.48 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 115 | 37 | 78 | 32.17% | 32.17% | 32.17% | 17.83 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 115 | 57 | 58 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 5 | -0.20 |
| BTC Hourly | transformer | Transformer | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 5 | -1.40 |
| BTC Hourly | nn | NN | 115 | 53 | 62 | 46.09% | 46.09% | 46.09% | 3.91 pp | -9 | 5 | -1.80 |
| BTC Hourly | rf | RandomForest | 115 | 49 | 66 | 42.61% | 42.61% | 42.61% | 7.39 pp | -17 | 5 | -3.40 |
| BTC Hourly | xgb | XGBoost | 115 | 42 | 73 | 36.52% | 36.52% | 36.52% | 13.48 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 115 | 37 | 78 | 32.17% | 32.17% | 32.17% | 17.83 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 141 | 68 | 73 | 48.23% | 48.23% | 48.23% | 1.77 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 141 | 58 | 83 | 41.13% | 41.13% | 41.13% | 8.87 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 151 | 54 | 97 | 35.76% | 35.76% | 35.76% | 14.24 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 141 | 49 | 92 | 34.75% | 34.75% | 34.75% | 15.25 pp | -43 | 7 | -6.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 139 | 72 | 67 | 51.80% | 51.80% | 51.80% | 1.80 pp | 5 | 11 | 0.45 |
| BTC Market Hours | rf | RandomForest | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| BTC Market Hours | transformer | Transformer | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 139 | 54 | 85 | 38.85% | 38.85% | 38.85% | 11.15 pp | -31 | 11 | -2.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 139 | 67 | 72 | 48.20% | 48.20% | 48.20% | 1.80 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | transformer | Transformer | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 12 | -0.75 |
| BTC Market Hours Daily | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 139 | 57 | 82 | 41.01% | 41.01% | 41.01% | 8.99 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 139 | 53 | 86 | 38.13% | 38.13% | 38.13% | 11.87 pp | -33 | 12 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
