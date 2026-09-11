# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T08:18:33.265443+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 530 | 283 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 284 | 145 | 139 | 51.06% | 50.42% | 51.06% | 1.06 pp | 6 | 22 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 283 | 140 | 143 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 283 | 137 | 146 | 48.41% | 49.58% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 283 | 137 | 146 | 48.41% | 48.33% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 260 | 125 | 135 | 48.08% | 47.92% | 48.08% | 1.92 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 284 | 132 | 152 | 46.48% | 46.67% | 46.48% | 3.52 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 284 | 131 | 153 | 46.13% | 46.25% | 46.13% | 3.87 pp | -22 | 22 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 284 | 128 | 156 | 45.07% | 45.83% | 45.07% | 4.93 pp | -28 | 22 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 284 | 126 | 158 | 44.37% | 43.33% | 44.37% | 5.63 pp | -32 | 22 | -1.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| BTC Market Hours Daily | rf | RandomForest | 283 | 122 | 161 | 43.11% | 43.33% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 283 | 122 | 161 | 43.11% | 42.92% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Daily | nn | NN | 286 | 131 | 155 | 45.80% | 45.00% | 45.80% | 4.20 pp | -24 | 13 | -1.85 |
| BTC Hourly | transformer | Transformer | 260 | 119 | 141 | 45.77% | 46.25% | 45.77% | 4.23 pp | -22 | 11 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 283 | 117 | 166 | 41.34% | 44.17% | 41.34% | 8.66 pp | -49 | 23 | -2.13 |
| BTC Daily | mlp_sklearn | MLPClassifier | 286 | 129 | 157 | 45.10% | 44.58% | 45.10% | 4.90 pp | -28 | 13 | -2.15 |
| BTC Market Hours | lstm | LSTM | 284 | 117 | 167 | 41.20% | 42.92% | 41.20% | 8.80 pp | -50 | 22 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 283 | 140 | 143 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 283 | 137 | 146 | 48.41% | 49.58% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 283 | 137 | 146 | 48.41% | 48.33% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | rf | RandomForest | 283 | 122 | 161 | 43.11% | 43.33% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 283 | 122 | 161 | 43.11% | 42.92% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 283 | 117 | 166 | 41.34% | 44.17% | 41.34% | 8.66 pp | -49 | 23 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
