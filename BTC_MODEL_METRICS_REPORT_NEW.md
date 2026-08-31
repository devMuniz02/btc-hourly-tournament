# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T10:11:27.423844+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 144 | 84 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 179 | 119 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 211 | 107 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 211 | 107 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| BTC Market Hours | nn | NN | 107 | 57 | 50 | 53.27% | 53.27% | 53.27% | 3.27 pp | 7 | 9 | 0.78 |
| BTC Hourly | transformer | Transformer | 84 | 43 | 41 | 51.19% | 51.19% | 51.19% | 1.19 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| BTC Hourly | nn | NN | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 107 | 53 | 54 | 49.53% | 49.53% | 49.53% | 0.47 pp | -1 | 10 | -0.10 |
| BTC Market Hours | rf | RandomForest | 107 | 51 | 56 | 47.66% | 47.66% | 47.66% | 2.34 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| BTC Market Hours Daily | nn | NN | 107 | 46 | 61 | 42.99% | 42.99% | 42.99% | 7.01 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | transformer | Transformer | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 9 | -2.11 |
| BTC Market Hours | lstm | LSTM | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 4 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 107 | 41 | 66 | 38.32% | 38.32% | 38.32% | 11.68 pp | -25 | 10 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 107 | 42 | 65 | 39.25% | 39.25% | 39.25% | 10.75 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 5 | -2.60 |
| BTC Market Hours Daily | xgb | XGBoost | 107 | 40 | 67 | 37.38% | 37.38% | 37.38% | 12.62 pp | -27 | 10 | -2.70 |
| BTC Hourly | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 119 | 42 | 77 | 35.29% | 35.29% | 35.29% | 14.71 pp | -35 | 6 | -5.83 |
| BTC Daily | lstm | LSTM | 109 | 39 | 70 | 35.78% | 35.78% | 35.78% | 14.22 pp | -31 | 5 | -6.20 |
| BTC Hourly | xgb | XGBoost | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 4 | -6.50 |
| BTC Hourly | lstm | LSTM | 84 | 28 | 56 | 33.33% | 33.33% | 33.33% | 16.67 pp | -28 | 4 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 84 | 43 | 41 | 51.19% | 51.19% | 51.19% | 1.19 pp | 2 | 4 | 0.50 |
| BTC Hourly | nn | NN | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 4 | -2.50 |
| BTC Hourly | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 4 | -3.50 |
| BTC Hourly | xgb | XGBoost | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 4 | -6.50 |
| BTC Hourly | lstm | LSTM | 84 | 28 | 56 | 33.33% | 33.33% | 33.33% | 16.67 pp | -28 | 4 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 119 | 42 | 77 | 35.29% | 35.29% | 35.29% | 14.71 pp | -35 | 6 | -5.83 |
| BTC Daily | lstm | LSTM | 109 | 39 | 70 | 35.78% | 35.78% | 35.78% | 14.22 pp | -31 | 5 | -6.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 107 | 57 | 50 | 53.27% | 53.27% | 53.27% | 3.27 pp | 7 | 9 | 0.78 |
| BTC Market Hours | rf | RandomForest | 107 | 51 | 56 | 47.66% | 47.66% | 47.66% | 2.34 pp | -5 | 9 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 9 | -2.11 |
| BTC Market Hours | lstm | LSTM | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 107 | 42 | 65 | 39.25% | 39.25% | 39.25% | 10.75 pp | -23 | 9 | -2.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 107 | 53 | 54 | 49.53% | 49.53% | 49.53% | 0.47 pp | -1 | 10 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 107 | 46 | 61 | 42.99% | 42.99% | 42.99% | 7.01 pp | -15 | 10 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 107 | 41 | 66 | 38.32% | 38.32% | 38.32% | 11.68 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 107 | 40 | 67 | 37.38% | 37.38% | 37.38% | 12.62 pp | -27 | 10 | -2.70 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
