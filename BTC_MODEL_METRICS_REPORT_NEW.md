# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T21:18:38.085303+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 20:00:00+00:00 | 374 | 196 | 178 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 20:00:00+00:00 | 374 | 196 | 178 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 169 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 169 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 47 | 122 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 47 | 122 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 196 | 103 | 93 | 52.55% | 52.55% | 52.55% | 2.55 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 196 | 101 | 95 | 51.53% | 51.53% | 51.53% | 1.53 pp | 6 | 16 | 0.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 172 | 87 | 85 | 50.58% | 50.58% | 50.58% | 0.58 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | transformer | Transformer | 196 | 97 | 99 | 49.49% | 49.49% | 49.49% | 0.51 pp | -2 | 16 | -0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 16 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 196 | 92 | 104 | 46.94% | 46.94% | 46.94% | 3.06 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 16 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 172 | 81 | 91 | 47.09% | 47.09% | 47.09% | 2.91 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 16 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 196 | 86 | 110 | 43.88% | 43.88% | 43.88% | 6.12 pp | -24 | 16 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| BTC Market Hours | lstm | LSTM | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 16 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 196 | 81 | 115 | 41.33% | 41.33% | 41.33% | 8.67 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 196 | 81 | 115 | 41.33% | 41.33% | 41.33% | 8.67 pp | -34 | 16 | -2.12 |
| BTC Daily | nn | NN | 198 | 89 | 109 | 44.95% | 44.95% | 44.95% | 5.05 pp | -20 | 9 | -2.22 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | rf | RandomForest | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| BTC Daily | transformer | Transformer | 198 | 84 | 114 | 42.42% | 42.42% | 42.42% | 7.58 pp | -30 | 9 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |
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
| BTC Market Hours Daily | transformer | Transformer | 196 | 103 | 93 | 52.55% | 52.55% | 52.55% | 2.55 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 16 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 196 | 81 | 115 | 41.33% | 41.33% | 41.33% | 8.67 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 196 | 81 | 115 | 41.33% | 41.33% | 41.33% | 8.67 pp | -34 | 16 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
