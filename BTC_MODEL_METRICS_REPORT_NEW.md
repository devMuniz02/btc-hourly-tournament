# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T06:01:36.960479+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 318 | 258 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 354 | 294 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 529 | 282 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 529 | 282 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T22:00:00+00:00 | 249 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T22:00:00+00:00 | 249 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T22:00:00+00:00 | 249 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T22:00:00+00:00 | 250 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 282 | 144 | 138 | 51.06% | 50.00% | 51.06% | 1.06 pp | 6 | 22 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 282 | 139 | 143 | 49.29% | 48.75% | 49.29% | 0.71 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 282 | 136 | 146 | 48.23% | 49.17% | 48.23% | 1.77 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | transformer | Transformer | 282 | 136 | 146 | 48.23% | 48.33% | 48.23% | 1.77 pp | -10 | 23 | -0.43 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 258 | 124 | 134 | 48.06% | 47.92% | 48.06% | 1.94 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 282 | 131 | 151 | 46.45% | 46.67% | 46.45% | 3.55 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 282 | 131 | 151 | 46.45% | 46.25% | 46.45% | 3.55 pp | -20 | 22 | -0.91 |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 249 | 116 | 133 | 46.59% | 46.67% | 46.59% | 3.41 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 116 | 133 | 46.59% | 46.67% | 46.59% | 3.41 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 282 | 127 | 155 | 45.04% | 45.42% | 45.04% | 4.96 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 282 | 125 | 157 | 44.33% | 43.33% | 44.33% | 5.67 pp | -32 | 22 | -1.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | lstm | LSTM | 249 | 112 | 137 | 44.98% | 44.17% | 44.98% | 5.02 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 112 | 137 | 44.98% | 44.17% | 44.98% | 5.02 pp | -25 | 15 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 282 | 121 | 161 | 42.91% | 42.92% | 42.91% | 7.09 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | xgb | XGBoost | 282 | 121 | 161 | 42.91% | 42.50% | 42.91% | 7.09 pp | -40 | 23 | -1.74 |
| Consolidated Hourly | transformer | Transformer | 249 | 111 | 138 | 44.58% | 44.17% | 44.58% | 5.42 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 111 | 138 | 44.58% | 44.17% | 44.58% | 5.42 pp | -27 | 15 | -1.80 |
| BTC Daily | nn | NN | 284 | 130 | 154 | 45.77% | 45.00% | 45.77% | 4.23 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 284 | 128 | 156 | 45.07% | 44.58% | 45.07% | 4.93 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 282 | 116 | 166 | 41.13% | 43.75% | 41.13% | 8.87 pp | -50 | 23 | -2.17 |
| BTC Hourly | transformer | Transformer | 258 | 117 | 141 | 45.35% | 45.83% | 45.35% | 4.65 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 282 | 117 | 165 | 41.49% | 43.33% | 41.49% | 8.51 pp | -48 | 22 | -2.18 |
| Consolidated Hourly | xgb | XGBoost | 249 | 107 | 142 | 42.97% | 42.92% | 42.97% | 7.03 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 107 | 142 | 42.97% | 42.92% | 42.97% | 7.03 pp | -35 | 15 | -2.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| BTC Hourly | nn | NN | 258 | 109 | 149 | 42.25% | 42.50% | 42.25% | 7.75 pp | -40 | 11 | -3.64 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 91 | 32 | 59 | 35.16% | 35.16% | 35.16% | 14.84 pp | -27 | 7 | -3.86 |
| BTC Daily | transformer | Transformer | 284 | 115 | 169 | 40.49% | 38.33% | 40.49% | 9.51 pp | -54 | 13 | -4.15 |
| BTC Hourly | rf | RandomForest | 258 | 105 | 153 | 40.70% | 40.42% | 40.70% | 9.30 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 284 | 108 | 176 | 38.03% | 37.08% | 38.03% | 11.97 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 294 | 110 | 184 | 37.41% | 37.50% | 37.41% | 12.59 pp | -74 | 14 | -5.29 |
| BTC Hourly | lstm | LSTM | 258 | 95 | 163 | 36.82% | 36.25% | 36.82% | 13.18 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 284 | 101 | 183 | 35.56% | 35.83% | 35.56% | 14.44 pp | -82 | 13 | -6.31 |
| BTC Hourly | xgb | XGBoost | 258 | 89 | 169 | 34.50% | 34.58% | 34.50% | 15.50 pp | -80 | 11 | -7.27 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 258 | 124 | 134 | 48.06% | 47.92% | 48.06% | 1.94 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 258 | 117 | 141 | 45.35% | 45.83% | 45.35% | 4.65 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 258 | 109 | 149 | 42.25% | 42.50% | 42.25% | 7.75 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 258 | 105 | 153 | 40.70% | 40.42% | 40.70% | 9.30 pp | -48 | 11 | -4.36 |
| BTC Hourly | lstm | LSTM | 258 | 95 | 163 | 36.82% | 36.25% | 36.82% | 13.18 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 258 | 89 | 169 | 34.50% | 34.58% | 34.50% | 15.50 pp | -80 | 11 | -7.27 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 284 | 130 | 154 | 45.77% | 45.00% | 45.77% | 4.23 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 284 | 128 | 156 | 45.07% | 44.58% | 45.07% | 4.93 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 284 | 115 | 169 | 40.49% | 38.33% | 40.49% | 9.51 pp | -54 | 13 | -4.15 |
| BTC Daily | rf | RandomForest | 284 | 108 | 176 | 38.03% | 37.08% | 38.03% | 11.97 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 294 | 110 | 184 | 37.41% | 37.50% | 37.41% | 12.59 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 284 | 101 | 183 | 35.56% | 35.83% | 35.56% | 14.44 pp | -82 | 13 | -6.31 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 282 | 144 | 138 | 51.06% | 50.00% | 51.06% | 1.06 pp | 6 | 22 | 0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 282 | 131 | 151 | 46.45% | 46.67% | 46.45% | 3.55 pp | -20 | 22 | -0.91 |
| BTC Market Hours | transformer | Transformer | 282 | 131 | 151 | 46.45% | 46.25% | 46.45% | 3.55 pp | -20 | 22 | -0.91 |
| BTC Market Hours | xgb | XGBoost | 282 | 127 | 155 | 45.04% | 45.42% | 45.04% | 4.96 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 282 | 125 | 157 | 44.33% | 43.33% | 44.33% | 5.67 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 282 | 117 | 165 | 41.49% | 43.33% | 41.49% | 8.51 pp | -48 | 22 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 282 | 139 | 143 | 49.29% | 48.75% | 49.29% | 0.71 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | nn | NN | 282 | 136 | 146 | 48.23% | 49.17% | 48.23% | 1.77 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | transformer | Transformer | 282 | 136 | 146 | 48.23% | 48.33% | 48.23% | 1.77 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | rf | RandomForest | 282 | 121 | 161 | 42.91% | 42.92% | 42.91% | 7.09 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | xgb | XGBoost | 282 | 121 | 161 | 42.91% | 42.50% | 42.91% | 7.09 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 282 | 116 | 166 | 41.13% | 43.75% | 41.13% | 8.87 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 249 | 116 | 133 | 46.59% | 46.67% | 46.59% | 3.41 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | lstm | LSTM | 249 | 112 | 137 | 44.98% | 44.17% | 44.98% | 5.02 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 249 | 111 | 138 | 44.58% | 44.17% | 44.58% | 5.42 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 249 | 107 | 142 | 42.97% | 42.92% | 42.97% | 7.03 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 116 | 133 | 46.59% | 46.67% | 46.59% | 3.41 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 112 | 137 | 44.98% | 44.17% | 44.98% | 5.02 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 111 | 138 | 44.58% | 44.17% | 44.58% | 5.42 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 107 | 142 | 42.97% | 42.92% | 42.97% | 7.03 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 105 | 144 | 42.17% | 42.92% | 42.17% | 7.83 pp | -39 | 15 | -2.60 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 32 | 59 | 35.16% | 35.16% | 35.16% | 14.84 pp | -27 | 7 | -3.86 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
