# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T20:09:29.598614+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 232 | 172 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 268 | 208 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 19:00:00+00:00 | 373 | 196 | 177 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 19:00:00+00:00 | 372 | 195 | 177 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 195 | 102 | 93 | 52.31% | 52.31% | 52.31% | 2.31 pp | 9 | 16 | 0.56 |
| BTC Market Hours | nn | NN | 196 | 101 | 95 | 51.53% | 51.53% | 51.53% | 1.53 pp | 6 | 16 | 0.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 172 | 87 | 85 | 50.58% | 50.58% | 50.58% | 0.58 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | transformer | Transformer | 196 | 97 | 99 | 49.49% | 49.49% | 49.49% | 0.51 pp | -2 | 16 | -0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 196 | 92 | 104 | 46.94% | 46.94% | 46.94% | 3.06 pp | -12 | 16 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 16 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 172 | 81 | 91 | 47.09% | 47.09% | 47.09% | 2.91 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 16 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 196 | 86 | 110 | 43.88% | 43.88% | 43.88% | 6.12 pp | -24 | 16 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| BTC Market Hours | lstm | LSTM | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 16 | -1.75 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| BTC Daily | nn | NN | 198 | 89 | 109 | 44.95% | 44.95% | 44.95% | 5.05 pp | -20 | 9 | -2.22 |
| BTC Hourly | rf | RandomForest | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 198 | 84 | 114 | 42.42% | 42.42% | 42.42% | 7.58 pp | -30 | 9 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| BTC Daily | rf | RandomForest | 198 | 77 | 121 | 38.89% | 38.89% | 38.89% | 11.11 pp | -44 | 9 | -4.89 |
| BTC Hourly | lstm | LSTM | 172 | 63 | 109 | 36.63% | 36.63% | 36.63% | 13.37 pp | -46 | 8 | -5.75 |
| BTC Hourly | xgb | XGBoost | 172 | 62 | 110 | 36.05% | 36.05% | 36.05% | 13.95 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 208 | 74 | 134 | 35.58% | 35.58% | 35.58% | 14.42 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 198 | 67 | 131 | 33.84% | 33.84% | 33.84% | 16.16 pp | -64 | 9 | -7.11 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 172 | 87 | 85 | 50.58% | 50.58% | 50.58% | 0.58 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 172 | 81 | 91 | 47.09% | 47.09% | 47.09% | 2.91 pp | -10 | 8 | -1.25 |
| BTC Hourly | rf | RandomForest | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 8 | -3.25 |
| BTC Hourly | lstm | LSTM | 172 | 63 | 109 | 36.63% | 36.63% | 36.63% | 13.37 pp | -46 | 8 | -5.75 |
| BTC Hourly | xgb | XGBoost | 172 | 62 | 110 | 36.05% | 36.05% | 36.05% | 13.95 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 198 | 89 | 109 | 44.95% | 44.95% | 44.95% | 5.05 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 198 | 84 | 114 | 42.42% | 42.42% | 42.42% | 7.58 pp | -30 | 9 | -3.33 |
| BTC Daily | rf | RandomForest | 198 | 77 | 121 | 38.89% | 38.89% | 38.89% | 11.11 pp | -44 | 9 | -4.89 |
| BTC Daily | xgb | XGBoost | 208 | 74 | 134 | 35.58% | 35.58% | 35.58% | 14.42 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 198 | 67 | 131 | 33.84% | 33.84% | 33.84% | 16.16 pp | -64 | 9 | -7.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 196 | 101 | 95 | 51.53% | 51.53% | 51.53% | 1.53 pp | 6 | 16 | 0.38 |
| BTC Market Hours | transformer | Transformer | 196 | 97 | 99 | 49.49% | 49.49% | 49.49% | 0.51 pp | -2 | 16 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 196 | 92 | 104 | 46.94% | 46.94% | 46.94% | 3.06 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 16 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 196 | 86 | 110 | 43.88% | 43.88% | 43.88% | 6.12 pp | -24 | 16 | -1.50 |
| BTC Market Hours | lstm | LSTM | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 16 | -1.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 195 | 102 | 93 | 52.31% | 52.31% | 52.31% | 2.31 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | rf | RandomForest | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | lstm | LSTM | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
