# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T11:18:00.662545+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 322 | 262 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 358 | 298 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 533 | 286 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 533 | 286 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 286 | 147 | 139 | 51.40% | 50.83% | 51.40% | 1.40 pp | 8 | 22 | 0.36 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 286 | 141 | 145 | 49.30% | 49.17% | 49.30% | 0.70 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 286 | 139 | 147 | 48.60% | 50.00% | 48.60% | 1.40 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | transformer | Transformer | 286 | 138 | 148 | 48.25% | 48.33% | 48.25% | 1.75 pp | -10 | 23 | -0.43 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 262 | 127 | 135 | 48.47% | 47.92% | 48.47% | 1.53 pp | -8 | 11 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 286 | 133 | 153 | 46.50% | 46.67% | 46.50% | 3.50 pp | -20 | 22 | -0.91 |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | transformer | Transformer | 286 | 132 | 154 | 46.15% | 46.25% | 46.15% | 3.85 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 286 | 129 | 157 | 45.10% | 46.25% | 45.10% | 4.90 pp | -28 | 22 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| BTC Market Hours | rf | RandomForest | 286 | 126 | 160 | 44.06% | 42.92% | 44.06% | 5.94 pp | -34 | 22 | -1.55 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 286 | 124 | 162 | 43.36% | 43.75% | 43.36% | 6.64 pp | -38 | 23 | -1.65 |
| BTC Daily | nn | NN | 288 | 133 | 155 | 46.18% | 45.42% | 46.18% | 3.82 pp | -22 | 13 | -1.69 |
| BTC Market Hours Daily | rf | RandomForest | 286 | 123 | 163 | 43.01% | 43.33% | 43.01% | 6.99 pp | -40 | 23 | -1.74 |
| BTC Hourly | transformer | Transformer | 262 | 121 | 141 | 46.18% | 46.67% | 46.18% | 3.82 pp | -20 | 11 | -1.82 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 288 | 130 | 158 | 45.14% | 44.17% | 45.14% | 4.86 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 286 | 118 | 168 | 41.26% | 44.17% | 41.26% | 8.74 pp | -50 | 23 | -2.17 |
| BTC Market Hours | lstm | LSTM | 286 | 118 | 168 | 41.26% | 42.92% | 41.26% | 8.74 pp | -50 | 22 | -2.27 |
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
| BTC Hourly | nn | NN | 262 | 110 | 152 | 41.98% | 41.25% | 41.98% | 8.02 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 288 | 116 | 172 | 40.28% | 37.50% | 40.28% | 9.72 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 262 | 105 | 157 | 40.08% | 40.42% | 40.08% | 9.92 pp | -52 | 11 | -4.73 |
| BTC Daily | xgb | XGBoost | 298 | 112 | 186 | 37.58% | 37.50% | 37.58% | 12.42 pp | -74 | 14 | -5.29 |
| BTC Daily | rf | RandomForest | 288 | 109 | 179 | 37.85% | 36.25% | 37.85% | 12.15 pp | -70 | 13 | -5.38 |
| BTC Daily | lstm | LSTM | 288 | 102 | 186 | 35.42% | 35.83% | 35.42% | 14.58 pp | -84 | 13 | -6.46 |
| BTC Hourly | lstm | LSTM | 262 | 95 | 167 | 36.26% | 35.00% | 36.26% | 13.74 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 262 | 90 | 172 | 34.35% | 34.58% | 34.35% | 15.65 pp | -82 | 11 | -7.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 262 | 127 | 135 | 48.47% | 47.92% | 48.47% | 1.53 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 262 | 121 | 141 | 46.18% | 46.67% | 46.18% | 3.82 pp | -20 | 11 | -1.82 |
| BTC Hourly | nn | NN | 262 | 110 | 152 | 41.98% | 41.25% | 41.98% | 8.02 pp | -42 | 11 | -3.82 |
| BTC Hourly | rf | RandomForest | 262 | 105 | 157 | 40.08% | 40.42% | 40.08% | 9.92 pp | -52 | 11 | -4.73 |
| BTC Hourly | lstm | LSTM | 262 | 95 | 167 | 36.26% | 35.00% | 36.26% | 13.74 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 262 | 90 | 172 | 34.35% | 34.58% | 34.35% | 15.65 pp | -82 | 11 | -7.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 288 | 133 | 155 | 46.18% | 45.42% | 46.18% | 3.82 pp | -22 | 13 | -1.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 288 | 130 | 158 | 45.14% | 44.17% | 45.14% | 4.86 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 288 | 116 | 172 | 40.28% | 37.50% | 40.28% | 9.72 pp | -56 | 13 | -4.31 |
| BTC Daily | xgb | XGBoost | 298 | 112 | 186 | 37.58% | 37.50% | 37.58% | 12.42 pp | -74 | 14 | -5.29 |
| BTC Daily | rf | RandomForest | 288 | 109 | 179 | 37.85% | 36.25% | 37.85% | 12.15 pp | -70 | 13 | -5.38 |
| BTC Daily | lstm | LSTM | 288 | 102 | 186 | 35.42% | 35.83% | 35.42% | 14.58 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 286 | 147 | 139 | 51.40% | 50.83% | 51.40% | 1.40 pp | 8 | 22 | 0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 286 | 133 | 153 | 46.50% | 46.67% | 46.50% | 3.50 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 286 | 132 | 154 | 46.15% | 46.25% | 46.15% | 3.85 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 286 | 129 | 157 | 45.10% | 46.25% | 45.10% | 4.90 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 286 | 126 | 160 | 44.06% | 42.92% | 44.06% | 5.94 pp | -34 | 22 | -1.55 |
| BTC Market Hours | lstm | LSTM | 286 | 118 | 168 | 41.26% | 42.92% | 41.26% | 8.74 pp | -50 | 22 | -2.27 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 286 | 141 | 145 | 49.30% | 49.17% | 49.30% | 0.70 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 286 | 139 | 147 | 48.60% | 50.00% | 48.60% | 1.40 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | transformer | Transformer | 286 | 138 | 148 | 48.25% | 48.33% | 48.25% | 1.75 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | xgb | XGBoost | 286 | 124 | 162 | 43.36% | 43.75% | 43.36% | 6.64 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | rf | RandomForest | 286 | 123 | 163 | 43.01% | 43.33% | 43.01% | 6.99 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 286 | 118 | 168 | 41.26% | 44.17% | 41.26% | 8.74 pp | -50 | 23 | -2.17 |

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
