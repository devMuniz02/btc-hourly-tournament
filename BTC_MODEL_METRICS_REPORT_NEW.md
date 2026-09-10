# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T05:47:44.125955+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 302 | 242 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 338 | 278 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 500 | 266 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 500 | 266 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 233 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 233 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 233 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 234 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 266 | 140 | 126 | 52.63% | 52.50% | 52.63% | 2.63 pp | 14 | 21 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 266 | 132 | 134 | 49.62% | 48.75% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 266 | 129 | 137 | 48.50% | 48.33% | 48.50% | 1.50 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 266 | 128 | 138 | 48.12% | 48.33% | 48.12% | 1.88 pp | -10 | 22 | -0.45 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 49.17% | 48.76% | 1.24 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | rf | RandomForest | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 15 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 266 | 125 | 141 | 46.99% | 47.50% | 46.99% | 3.01 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 266 | 124 | 142 | 46.62% | 46.67% | 46.62% | 3.38 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 266 | 123 | 143 | 46.24% | 45.00% | 46.24% | 3.76 pp | -20 | 21 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 266 | 117 | 149 | 43.98% | 42.92% | 43.98% | 6.02 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 266 | 117 | 149 | 43.98% | 42.50% | 43.98% | 6.02 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 266 | 114 | 152 | 42.86% | 41.67% | 42.86% | 7.14 pp | -38 | 22 | -1.73 |
| Consolidated Hourly | xgb | XGBoost | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 15 | -1.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 233 | 100 | 133 | 42.92% | 42.92% | 42.92% | 7.08 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 100 | 133 | 42.92% | 42.92% | 42.92% | 7.08 pp | -33 | 15 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 266 | 108 | 158 | 40.60% | 41.67% | 40.60% | 9.40 pp | -50 | 22 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Hourly | transformer | Transformer | 242 | 108 | 134 | 44.63% | 45.00% | 44.63% | 5.37 pp | -26 | 11 | -2.36 |
| BTC Market Hours | lstm | LSTM | 266 | 108 | 158 | 40.60% | 42.50% | 40.60% | 9.40 pp | -50 | 21 | -2.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 15 | -2.73 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 242 | 102 | 140 | 42.15% | 42.50% | 42.15% | 7.85 pp | -38 | 11 | -3.45 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 242 | 99 | 143 | 40.91% | 41.25% | 40.91% | 9.09 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 268 | 109 | 159 | 40.67% | 39.17% | 40.67% | 9.33 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 268 | 102 | 166 | 38.06% | 37.92% | 38.06% | 11.94 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 278 | 103 | 175 | 37.05% | 37.50% | 37.05% | 12.95 pp | -72 | 13 | -5.54 |
| BTC Hourly | lstm | LSTM | 242 | 90 | 152 | 37.19% | 37.50% | 37.19% | 12.81 pp | -62 | 11 | -5.64 |
| BTC Daily | lstm | LSTM | 268 | 96 | 172 | 35.82% | 36.67% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 242 | 84 | 158 | 34.71% | 35.00% | 34.71% | 15.29 pp | -74 | 11 | -6.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 49.17% | 48.76% | 1.24 pp | -6 | 11 | -0.55 |
| BTC Hourly | transformer | Transformer | 242 | 108 | 134 | 44.63% | 45.00% | 44.63% | 5.37 pp | -26 | 11 | -2.36 |
| BTC Hourly | nn | NN | 242 | 102 | 140 | 42.15% | 42.50% | 42.15% | 7.85 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 242 | 99 | 143 | 40.91% | 41.25% | 40.91% | 9.09 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 242 | 90 | 152 | 37.19% | 37.50% | 37.19% | 12.81 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 242 | 84 | 158 | 34.71% | 35.00% | 34.71% | 15.29 pp | -74 | 11 | -6.73 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | transformer | Transformer | 268 | 109 | 159 | 40.67% | 39.17% | 40.67% | 9.33 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 268 | 102 | 166 | 38.06% | 37.92% | 38.06% | 11.94 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 278 | 103 | 175 | 37.05% | 37.50% | 37.05% | 12.95 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 268 | 96 | 172 | 35.82% | 36.67% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 266 | 140 | 126 | 52.63% | 52.50% | 52.63% | 2.63 pp | 14 | 21 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 266 | 125 | 141 | 46.99% | 47.50% | 46.99% | 3.01 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 266 | 124 | 142 | 46.62% | 46.67% | 46.62% | 3.38 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 266 | 123 | 143 | 46.24% | 45.00% | 46.24% | 3.76 pp | -20 | 21 | -0.95 |
| BTC Market Hours | rf | RandomForest | 266 | 117 | 149 | 43.98% | 42.50% | 43.98% | 6.02 pp | -32 | 21 | -1.52 |
| BTC Market Hours | lstm | LSTM | 266 | 108 | 158 | 40.60% | 42.50% | 40.60% | 9.40 pp | -50 | 21 | -2.38 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 266 | 132 | 134 | 49.62% | 48.75% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 266 | 129 | 137 | 48.50% | 48.33% | 48.50% | 1.50 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 266 | 128 | 138 | 48.12% | 48.33% | 48.12% | 1.88 pp | -10 | 22 | -0.45 |
| BTC Market Hours Daily | xgb | XGBoost | 266 | 117 | 149 | 43.98% | 42.92% | 43.98% | 6.02 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 266 | 114 | 152 | 42.86% | 41.67% | 42.86% | 7.14 pp | -38 | 22 | -1.73 |
| BTC Market Hours Daily | lstm | LSTM | 266 | 108 | 158 | 40.60% | 41.67% | 40.60% | 9.40 pp | -50 | 22 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | xgb | XGBoost | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 233 | 100 | 133 | 42.92% | 42.92% | 42.92% | 7.08 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 15 | -2.73 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 100 | 133 | 42.92% | 42.92% | 42.92% | 7.08 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 15 | -2.73 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
