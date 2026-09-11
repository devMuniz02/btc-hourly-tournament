# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T08:51:24.321888+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 320 | 260 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 356 | 296 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 531 | 284 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 531 | 284 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 251 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 284 | 145 | 139 | 51.06% | 50.42% | 51.06% | 1.06 pp | 6 | 22 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 284 | 140 | 144 | 49.30% | 49.17% | 49.30% | 0.70 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 284 | 138 | 146 | 48.59% | 50.00% | 48.59% | 1.41 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | transformer | Transformer | 284 | 137 | 147 | 48.24% | 48.33% | 48.24% | 1.76 pp | -10 | 23 | -0.43 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 260 | 125 | 135 | 48.08% | 47.92% | 48.08% | 1.92 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 284 | 132 | 152 | 46.48% | 46.67% | 46.48% | 3.52 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 284 | 131 | 153 | 46.13% | 46.25% | 46.13% | 3.87 pp | -22 | 22 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 42 | 50 | 45.65% | 45.65% | 45.65% | 4.35 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| BTC Market Hours | xgb | XGBoost | 284 | 128 | 156 | 45.07% | 45.83% | 45.07% | 4.93 pp | -28 | 22 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 284 | 126 | 158 | 44.37% | 43.33% | 44.37% | 5.63 pp | -32 | 22 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 284 | 123 | 161 | 43.31% | 43.75% | 43.31% | 6.69 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | xgb | XGBoost | 284 | 123 | 161 | 43.31% | 43.33% | 43.31% | 6.69 pp | -38 | 23 | -1.65 |
| BTC Daily | nn | NN | 286 | 131 | 155 | 45.80% | 45.00% | 45.80% | 4.20 pp | -24 | 13 | -1.85 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| BTC Hourly | transformer | Transformer | 260 | 119 | 141 | 45.77% | 46.25% | 45.77% | 4.23 pp | -22 | 11 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 7 | -2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 286 | 129 | 157 | 45.10% | 44.58% | 45.10% | 4.90 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 284 | 117 | 167 | 41.20% | 44.17% | 41.20% | 8.80 pp | -50 | 23 | -2.17 |
| BTC Market Hours | lstm | LSTM | 284 | 117 | 167 | 41.20% | 42.92% | 41.20% | 8.80 pp | -50 | 22 | -2.27 |
| Consolidated Hourly | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Hourly | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| BTC Hourly | nn | NN | 260 | 109 | 151 | 41.92% | 41.67% | 41.92% | 8.08 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 286 | 115 | 171 | 40.21% | 37.92% | 40.21% | 9.79 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 260 | 105 | 155 | 40.38% | 40.42% | 40.38% | 9.62 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 286 | 109 | 177 | 38.11% | 37.08% | 38.11% | 11.89 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 296 | 111 | 185 | 37.50% | 37.50% | 37.50% | 12.50 pp | -74 | 14 | -5.29 |
| BTC Hourly | lstm | LSTM | 260 | 95 | 165 | 36.54% | 35.42% | 36.54% | 13.46 pp | -70 | 11 | -6.36 |
| BTC Daily | lstm | LSTM | 286 | 101 | 185 | 35.31% | 35.83% | 35.31% | 14.69 pp | -84 | 13 | -6.46 |
| BTC Hourly | xgb | XGBoost | 260 | 89 | 171 | 34.23% | 34.58% | 34.23% | 15.77 pp | -82 | 11 | -7.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 260 | 125 | 135 | 48.08% | 47.92% | 48.08% | 1.92 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 260 | 119 | 141 | 45.77% | 46.25% | 45.77% | 4.23 pp | -22 | 11 | -2.00 |
| BTC Hourly | nn | NN | 260 | 109 | 151 | 41.92% | 41.67% | 41.92% | 8.08 pp | -42 | 11 | -3.82 |
| BTC Hourly | rf | RandomForest | 260 | 105 | 155 | 40.38% | 40.42% | 40.38% | 9.62 pp | -50 | 11 | -4.55 |
| BTC Hourly | lstm | LSTM | 260 | 95 | 165 | 36.54% | 35.42% | 36.54% | 13.46 pp | -70 | 11 | -6.36 |
| BTC Hourly | xgb | XGBoost | 260 | 89 | 171 | 34.23% | 34.58% | 34.23% | 15.77 pp | -82 | 11 | -7.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 286 | 131 | 155 | 45.80% | 45.00% | 45.80% | 4.20 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 286 | 129 | 157 | 45.10% | 44.58% | 45.10% | 4.90 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 286 | 115 | 171 | 40.21% | 37.92% | 40.21% | 9.79 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 286 | 109 | 177 | 38.11% | 37.08% | 38.11% | 11.89 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 296 | 111 | 185 | 37.50% | 37.50% | 37.50% | 12.50 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 286 | 101 | 185 | 35.31% | 35.83% | 35.31% | 14.69 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 284 | 145 | 139 | 51.06% | 50.42% | 51.06% | 1.06 pp | 6 | 22 | 0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 284 | 132 | 152 | 46.48% | 46.67% | 46.48% | 3.52 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 284 | 131 | 153 | 46.13% | 46.25% | 46.13% | 3.87 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 284 | 128 | 156 | 45.07% | 45.83% | 45.07% | 4.93 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 284 | 126 | 158 | 44.37% | 43.33% | 44.37% | 5.63 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 284 | 117 | 167 | 41.20% | 42.92% | 41.20% | 8.80 pp | -50 | 22 | -2.27 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 284 | 140 | 144 | 49.30% | 49.17% | 49.30% | 0.70 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 284 | 138 | 146 | 48.59% | 50.00% | 48.59% | 1.41 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | transformer | Transformer | 284 | 137 | 147 | 48.24% | 48.33% | 48.24% | 1.76 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | rf | RandomForest | 284 | 123 | 161 | 43.31% | 43.75% | 43.31% | 6.69 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | xgb | XGBoost | 284 | 123 | 161 | 43.31% | 43.33% | 43.31% | 6.69 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | lstm | LSTM | 284 | 117 | 167 | 41.20% | 44.17% | 41.20% | 8.80 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Hourly | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 42 | 50 | 45.65% | 45.65% | 45.65% | 4.35 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
