# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T10:55:41.015259+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 321 | 261 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 357 | 297 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 532 | 285 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 532 | 285 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 285 | 146 | 139 | 51.23% | 50.83% | 51.23% | 1.23 pp | 7 | 22 | 0.32 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 285 | 140 | 145 | 49.12% | 48.75% | 49.12% | 0.88 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | nn | NN | 285 | 138 | 147 | 48.42% | 49.58% | 48.42% | 1.58 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 285 | 137 | 148 | 48.07% | 48.33% | 48.07% | 1.93 pp | -11 | 23 | -0.48 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 261 | 127 | 134 | 48.66% | 47.92% | 48.66% | 1.34 pp | -7 | 11 | -0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 285 | 133 | 152 | 46.67% | 47.08% | 46.67% | 3.33 pp | -19 | 22 | -0.86 |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | transformer | Transformer | 285 | 132 | 153 | 46.32% | 46.67% | 46.32% | 3.68 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 285 | 129 | 156 | 45.26% | 46.25% | 45.26% | 4.74 pp | -27 | 22 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| BTC Market Hours | rf | RandomForest | 285 | 126 | 159 | 44.21% | 43.33% | 44.21% | 5.79 pp | -33 | 22 | -1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 285 | 123 | 162 | 43.16% | 43.33% | 43.16% | 6.84 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 285 | 123 | 162 | 43.16% | 43.33% | 43.16% | 6.84 pp | -39 | 23 | -1.70 |
| BTC Hourly | transformer | Transformer | 261 | 121 | 140 | 46.36% | 46.67% | 46.36% | 3.64 pp | -19 | 11 | -1.73 |
| BTC Daily | nn | NN | 287 | 132 | 155 | 45.99% | 45.00% | 45.99% | 4.01 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 287 | 130 | 157 | 45.30% | 44.58% | 45.30% | 4.70 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 285 | 117 | 168 | 41.05% | 43.75% | 41.05% | 8.95 pp | -51 | 23 | -2.22 |
| BTC Market Hours | lstm | LSTM | 285 | 118 | 167 | 41.40% | 43.33% | 41.40% | 8.60 pp | -49 | 22 | -2.23 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| BTC Hourly | nn | NN | 261 | 110 | 151 | 42.15% | 41.25% | 42.15% | 7.85 pp | -41 | 11 | -3.73 |
| BTC Daily | transformer | Transformer | 287 | 115 | 172 | 40.07% | 37.50% | 40.07% | 9.93 pp | -57 | 13 | -4.38 |
| BTC Hourly | rf | RandomForest | 261 | 105 | 156 | 40.23% | 40.42% | 40.23% | 9.77 pp | -51 | 11 | -4.64 |
| BTC Daily | rf | RandomForest | 287 | 109 | 178 | 37.98% | 36.67% | 37.98% | 12.02 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 297 | 111 | 186 | 37.37% | 37.50% | 37.37% | 12.63 pp | -75 | 14 | -5.36 |
| BTC Hourly | lstm | LSTM | 261 | 95 | 166 | 36.40% | 35.00% | 36.40% | 13.60 pp | -71 | 11 | -6.45 |
| BTC Daily | lstm | LSTM | 287 | 101 | 186 | 35.19% | 35.42% | 35.19% | 14.81 pp | -85 | 13 | -6.54 |
| BTC Hourly | xgb | XGBoost | 261 | 90 | 171 | 34.48% | 34.58% | 34.48% | 15.52 pp | -81 | 11 | -7.36 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 261 | 127 | 134 | 48.66% | 47.92% | 48.66% | 1.34 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 261 | 121 | 140 | 46.36% | 46.67% | 46.36% | 3.64 pp | -19 | 11 | -1.73 |
| BTC Hourly | nn | NN | 261 | 110 | 151 | 42.15% | 41.25% | 42.15% | 7.85 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 261 | 105 | 156 | 40.23% | 40.42% | 40.23% | 9.77 pp | -51 | 11 | -4.64 |
| BTC Hourly | lstm | LSTM | 261 | 95 | 166 | 36.40% | 35.00% | 36.40% | 13.60 pp | -71 | 11 | -6.45 |
| BTC Hourly | xgb | XGBoost | 261 | 90 | 171 | 34.48% | 34.58% | 34.48% | 15.52 pp | -81 | 11 | -7.36 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 287 | 132 | 155 | 45.99% | 45.00% | 45.99% | 4.01 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 287 | 130 | 157 | 45.30% | 44.58% | 45.30% | 4.70 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 287 | 115 | 172 | 40.07% | 37.50% | 40.07% | 9.93 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 287 | 109 | 178 | 37.98% | 36.67% | 37.98% | 12.02 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 297 | 111 | 186 | 37.37% | 37.50% | 37.37% | 12.63 pp | -75 | 14 | -5.36 |
| BTC Daily | lstm | LSTM | 287 | 101 | 186 | 35.19% | 35.42% | 35.19% | 14.81 pp | -85 | 13 | -6.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 285 | 146 | 139 | 51.23% | 50.83% | 51.23% | 1.23 pp | 7 | 22 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 285 | 133 | 152 | 46.67% | 47.08% | 46.67% | 3.33 pp | -19 | 22 | -0.86 |
| BTC Market Hours | transformer | Transformer | 285 | 132 | 153 | 46.32% | 46.67% | 46.32% | 3.68 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 285 | 129 | 156 | 45.26% | 46.25% | 45.26% | 4.74 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 285 | 126 | 159 | 44.21% | 43.33% | 44.21% | 5.79 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 285 | 118 | 167 | 41.40% | 43.33% | 41.40% | 8.60 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 285 | 140 | 145 | 49.12% | 48.75% | 49.12% | 0.88 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | nn | NN | 285 | 138 | 147 | 48.42% | 49.58% | 48.42% | 1.58 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 285 | 137 | 148 | 48.07% | 48.33% | 48.07% | 1.93 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | rf | RandomForest | 285 | 123 | 162 | 43.16% | 43.33% | 43.16% | 6.84 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 285 | 123 | 162 | 43.16% | 43.33% | 43.16% | 6.84 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 285 | 117 | 168 | 41.05% | 43.75% | 41.05% | 8.95 pp | -51 | 23 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
