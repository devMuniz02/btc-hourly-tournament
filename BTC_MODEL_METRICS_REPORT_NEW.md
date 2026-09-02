# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T14:50:26.660134+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 179 | 119 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 215 | 155 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 13:00:00+00:00 | 275 | 143 | 132 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 13:00:00+00:00 | 275 | 143 | 132 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 120 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| BTC Market Hours | nn | NN | 143 | 74 | 69 | 51.75% | 51.75% | 51.75% | 1.75 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 119 | 59 | 60 | 49.58% | 49.58% | 49.58% | 0.42 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 12 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 7 | -0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 143 | 68 | 75 | 47.55% | 47.55% | 47.55% | 2.45 pp | -7 | 12 | -0.58 |
| BTC Hourly | transformer | Transformer | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 5 | -0.60 |
| BTC Market Hours | rf | RandomForest | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| BTC Market Hours | transformer | Transformer | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | nn | NN | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| BTC Daily | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 143 | 58 | 85 | 40.56% | 40.56% | 40.56% | 9.44 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| BTC Market Hours | lstm | LSTM | 143 | 57 | 86 | 39.86% | 39.86% | 39.86% | 10.14 pp | -29 | 11 | -2.64 |
| BTC Daily | transformer | Transformer | 145 | 63 | 82 | 43.45% | 43.45% | 43.45% | 6.55 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 143 | 54 | 89 | 37.76% | 37.76% | 37.76% | 12.24 pp | -35 | 12 | -2.92 |
| BTC Daily | rf | RandomForest | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 145 | 52 | 93 | 35.86% | 35.86% | 35.86% | 14.14 pp | -41 | 7 | -5.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 5 | -6.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| BTC Hourly | lstm | LSTM | 119 | 39 | 80 | 32.77% | 32.77% | 32.77% | 17.23 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 119 | 59 | 60 | 49.58% | 49.58% | 49.58% | 0.42 pp | -1 | 5 | -0.20 |
| BTC Hourly | transformer | Transformer | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 5 | -0.60 |
| BTC Hourly | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 5 | -1.80 |
| BTC Hourly | rf | RandomForest | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 119 | 39 | 80 | 32.77% | 32.77% | 32.77% | 17.23 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 145 | 63 | 82 | 43.45% | 43.45% | 43.45% | 6.55 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 7 | -3.29 |
| BTC Daily | xgb | XGBoost | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 145 | 52 | 93 | 35.86% | 35.86% | 35.86% | 14.14 pp | -41 | 7 | -5.86 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 143 | 74 | 69 | 51.75% | 51.75% | 51.75% | 1.75 pp | 5 | 11 | 0.45 |
| BTC Market Hours | rf | RandomForest | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| BTC Market Hours | transformer | Transformer | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| BTC Market Hours | xgb | XGBoost | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| BTC Market Hours | lstm | LSTM | 143 | 57 | 86 | 39.86% | 39.86% | 39.86% | 10.14 pp | -29 | 11 | -2.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | transformer | Transformer | 143 | 68 | 75 | 47.55% | 47.55% | 47.55% | 2.45 pp | -7 | 12 | -0.58 |
| BTC Market Hours Daily | nn | NN | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | xgb | XGBoost | 143 | 58 | 85 | 40.56% | 40.56% | 40.56% | 9.44 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 143 | 54 | 89 | 37.76% | 37.76% | 37.76% | 12.24 pp | -35 | 12 | -2.92 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
