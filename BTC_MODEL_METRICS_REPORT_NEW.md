# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T18:03:30.691277+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 149 | 89 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 185 | 125 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 17:00:00+00:00 | 223 | 113 | 110 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 17:00:00+00:00 | 223 | 113 | 110 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| BTC Market Hours | nn | NN | 113 | 61 | 52 | 53.98% | 53.98% | 53.98% | 3.98 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 113 | 57 | 56 | 50.44% | 50.44% | 50.44% | 0.44 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 89 | 44 | 45 | 49.44% | 49.44% | 49.44% | 0.56 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 115 | 56 | 59 | 48.70% | 48.70% | 48.70% | 1.30 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| BTC Hourly | nn | NN | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 113 | 53 | 60 | 46.90% | 46.90% | 46.90% | 3.10 pp | -7 | 9 | -0.78 |
| BTC Market Hours | rf | RandomForest | 113 | 53 | 60 | 46.90% | 46.90% | 46.90% | 3.10 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 113 | 48 | 65 | 42.48% | 42.48% | 42.48% | 7.52 pp | -17 | 9 | -1.89 |
| BTC Daily | transformer | Transformer | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 10 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 113 | 45 | 68 | 39.82% | 39.82% | 39.82% | 10.18 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 9 | -2.78 |
| BTC Market Hours Daily | lstm | LSTM | 113 | 42 | 71 | 37.17% | 37.17% | 37.17% | 12.83 pp | -29 | 10 | -2.90 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 115 | 46 | 69 | 40.00% | 40.00% | 40.00% | 10.00 pp | -23 | 6 | -3.83 |
| BTC Hourly | rf | RandomForest | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 125 | 45 | 80 | 36.00% | 36.00% | 36.00% | 14.00 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 115 | 40 | 75 | 34.78% | 34.78% | 34.78% | 15.22 pp | -35 | 6 | -5.83 |
| BTC Hourly | xgb | XGBoost | 89 | 30 | 59 | 33.71% | 33.71% | 33.71% | 16.29 pp | -29 | 4 | -7.25 |
| BTC Hourly | lstm | LSTM | 89 | 29 | 60 | 32.58% | 32.58% | 32.58% | 17.42 pp | -31 | 4 | -7.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 89 | 44 | 45 | 49.44% | 49.44% | 49.44% | 0.56 pp | -1 | 4 | -0.25 |
| BTC Hourly | nn | NN | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 4 | -4.25 |
| BTC Hourly | xgb | XGBoost | 89 | 30 | 59 | 33.71% | 33.71% | 33.71% | 16.29 pp | -29 | 4 | -7.25 |
| BTC Hourly | lstm | LSTM | 89 | 29 | 60 | 32.58% | 32.58% | 32.58% | 17.42 pp | -31 | 4 | -7.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 115 | 56 | 59 | 48.70% | 48.70% | 48.70% | 1.30 pp | -3 | 6 | -0.50 |
| BTC Daily | nn | NN | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 6 | -1.17 |
| BTC Daily | transformer | Transformer | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 6 | -2.17 |
| BTC Daily | rf | RandomForest | 115 | 46 | 69 | 40.00% | 40.00% | 40.00% | 10.00 pp | -23 | 6 | -3.83 |
| BTC Daily | xgb | XGBoost | 125 | 45 | 80 | 36.00% | 36.00% | 36.00% | 14.00 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 115 | 40 | 75 | 34.78% | 34.78% | 34.78% | 15.22 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 113 | 61 | 52 | 53.98% | 53.98% | 53.98% | 3.98 pp | 9 | 9 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 113 | 53 | 60 | 46.90% | 46.90% | 46.90% | 3.10 pp | -7 | 9 | -0.78 |
| BTC Market Hours | rf | RandomForest | 113 | 53 | 60 | 46.90% | 46.90% | 46.90% | 3.10 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 113 | 48 | 65 | 42.48% | 42.48% | 42.48% | 7.52 pp | -17 | 9 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 113 | 45 | 68 | 39.82% | 39.82% | 39.82% | 10.18 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 9 | -2.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 113 | 57 | 56 | 50.44% | 50.44% | 50.44% | 0.44 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | nn | NN | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 113 | 42 | 71 | 37.17% | 37.17% | 37.17% | 12.83 pp | -29 | 10 | -2.90 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
