# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T20:04:17.217645+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 167 | 107 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 203 | 143 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 19:00:00+00:00 | 256 | 131 | 125 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 19:00:00+00:00 | 256 | 131 | 125 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| BTC Market Hours | nn | NN | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| BTC Market Hours | rf | RandomForest | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 6 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 131 | 62 | 69 | 47.33% | 47.33% | 47.33% | 2.67 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Hourly | nn | NN | 107 | 51 | 56 | 47.66% | 47.66% | 47.66% | 2.34 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 11 | -1.73 |
| BTC Daily | nn | NN | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 131 | 55 | 76 | 41.98% | 41.98% | 41.98% | 8.02 pp | -21 | 11 | -1.91 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 131 | 53 | 78 | 40.46% | 40.46% | 40.46% | 9.54 pp | -25 | 11 | -2.27 |
| BTC Daily | transformer | Transformer | 133 | 59 | 74 | 44.36% | 44.36% | 44.36% | 5.64 pp | -15 | 6 | -2.50 |
| BTC Market Hours | lstm | LSTM | 131 | 50 | 81 | 38.17% | 38.17% | 38.17% | 11.83 pp | -31 | 11 | -2.82 |
| BTC Market Hours Daily | lstm | LSTM | 131 | 50 | 81 | 38.17% | 38.17% | 38.17% | 11.83 pp | -31 | 11 | -2.82 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 6 | -3.83 |
| BTC Hourly | rf | RandomForest | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 143 | 51 | 92 | 35.66% | 35.66% | 35.66% | 14.34 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 107 | 38 | 69 | 35.51% | 35.51% | 35.51% | 14.49 pp | -31 | 5 | -6.20 |
| BTC Daily | lstm | LSTM | 133 | 46 | 87 | 34.59% | 34.59% | 34.59% | 15.41 pp | -41 | 6 | -6.83 |
| BTC Hourly | lstm | LSTM | 107 | 34 | 73 | 31.78% | 31.78% | 31.78% | 18.22 pp | -39 | 5 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 5 | -0.60 |
| BTC Hourly | nn | NN | 107 | 51 | 56 | 47.66% | 47.66% | 47.66% | 2.34 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 5 | -1.40 |
| BTC Hourly | rf | RandomForest | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 5 | -4.20 |
| BTC Hourly | xgb | XGBoost | 107 | 38 | 69 | 35.51% | 35.51% | 35.51% | 14.49 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 107 | 34 | 73 | 31.78% | 31.78% | 31.78% | 18.22 pp | -39 | 5 | -7.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 6 | -0.50 |
| BTC Daily | nn | NN | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 6 | -1.83 |
| BTC Daily | transformer | Transformer | 133 | 59 | 74 | 44.36% | 44.36% | 44.36% | 5.64 pp | -15 | 6 | -2.50 |
| BTC Daily | rf | RandomForest | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 6 | -3.83 |
| BTC Daily | xgb | XGBoost | 143 | 51 | 92 | 35.66% | 35.66% | 35.66% | 14.34 pp | -41 | 7 | -5.86 |
| BTC Daily | lstm | LSTM | 133 | 46 | 87 | 34.59% | 34.59% | 34.59% | 15.41 pp | -41 | 6 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| BTC Market Hours | rf | RandomForest | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours | transformer | Transformer | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 131 | 55 | 76 | 41.98% | 41.98% | 41.98% | 8.02 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 131 | 50 | 81 | 38.17% | 38.17% | 38.17% | 11.83 pp | -31 | 11 | -2.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 131 | 62 | 69 | 47.33% | 47.33% | 47.33% | 2.67 pp | -7 | 11 | -0.64 |
| BTC Market Hours Daily | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | xgb | XGBoost | 131 | 53 | 78 | 40.46% | 40.46% | 40.46% | 9.54 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 131 | 50 | 81 | 38.17% | 38.17% | 38.17% | 11.83 pp | -31 | 11 | -2.82 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
