# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T21:12:43.402056+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 313 | 253 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 348 | 288 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 519 | 276 | 243 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 519 | 276 | 243 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 276 | 141 | 135 | 51.09% | 51.25% | 51.09% | 1.09 pp | 6 | 22 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 276 | 135 | 141 | 48.91% | 48.33% | 48.91% | 1.09 pp | -6 | 23 | -0.26 |
| BTC Market Hours Daily | transformer | Transformer | 276 | 134 | 142 | 48.55% | 47.92% | 48.55% | 1.45 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | nn | NN | 276 | 132 | 144 | 47.83% | 48.33% | 47.83% | 2.17 pp | -12 | 23 | -0.52 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 253 | 123 | 130 | 48.62% | 48.33% | 48.62% | 1.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| BTC Market Hours | transformer | Transformer | 276 | 128 | 148 | 46.38% | 45.83% | 46.38% | 3.62 pp | -20 | 22 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 276 | 127 | 149 | 46.01% | 46.25% | 46.01% | 3.99 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| BTC Market Hours | xgb | XGBoost | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 22 | -1.18 |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 276 | 122 | 154 | 44.20% | 42.92% | 44.20% | 5.80 pp | -32 | 22 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 276 | 119 | 157 | 43.12% | 42.50% | 43.12% | 6.88 pp | -38 | 23 | -1.65 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 276 | 118 | 158 | 42.75% | 42.08% | 42.75% | 7.25 pp | -40 | 23 | -1.74 |
| BTC Daily | nn | NN | 278 | 127 | 151 | 45.68% | 45.00% | 45.68% | 4.32 pp | -24 | 12 | -2.00 |
| BTC Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 46.25% | 45.45% | 4.55 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 276 | 113 | 163 | 40.94% | 42.50% | 40.94% | 9.06 pp | -50 | 23 | -2.17 |
| BTC Market Hours | lstm | LSTM | 276 | 114 | 162 | 41.30% | 42.50% | 41.30% | 8.70 pp | -48 | 22 | -2.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 278 | 125 | 153 | 44.96% | 44.58% | 44.96% | 5.04 pp | -28 | 12 | -2.33 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| BTC Hourly | nn | NN | 253 | 106 | 147 | 41.90% | 42.08% | 41.90% | 8.10 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 253 | 104 | 149 | 41.11% | 41.67% | 41.11% | 8.89 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 278 | 112 | 166 | 40.29% | 37.92% | 40.29% | 9.71 pp | -54 | 12 | -4.50 |
| BTC Daily | xgb | XGBoost | 288 | 107 | 181 | 37.15% | 37.50% | 37.15% | 12.85 pp | -74 | 13 | -5.69 |
| BTC Daily | rf | RandomForest | 278 | 104 | 174 | 37.41% | 36.67% | 37.41% | 12.59 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 253 | 93 | 160 | 36.76% | 36.25% | 36.76% | 13.24 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 278 | 98 | 180 | 35.25% | 35.42% | 35.25% | 14.75 pp | -82 | 12 | -6.83 |
| BTC Hourly | xgb | XGBoost | 253 | 88 | 165 | 34.78% | 35.42% | 34.78% | 15.22 pp | -77 | 11 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 253 | 123 | 130 | 48.62% | 48.33% | 48.62% | 1.38 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 46.25% | 45.45% | 4.55 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 253 | 106 | 147 | 41.90% | 42.08% | 41.90% | 8.10 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 253 | 104 | 149 | 41.11% | 41.67% | 41.11% | 8.89 pp | -45 | 11 | -4.09 |
| BTC Hourly | lstm | LSTM | 253 | 93 | 160 | 36.76% | 36.25% | 36.76% | 13.24 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 253 | 88 | 165 | 34.78% | 35.42% | 34.78% | 15.22 pp | -77 | 11 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 278 | 127 | 151 | 45.68% | 45.00% | 45.68% | 4.32 pp | -24 | 12 | -2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 278 | 125 | 153 | 44.96% | 44.58% | 44.96% | 5.04 pp | -28 | 12 | -2.33 |
| BTC Daily | transformer | Transformer | 278 | 112 | 166 | 40.29% | 37.92% | 40.29% | 9.71 pp | -54 | 12 | -4.50 |
| BTC Daily | xgb | XGBoost | 288 | 107 | 181 | 37.15% | 37.50% | 37.15% | 12.85 pp | -74 | 13 | -5.69 |
| BTC Daily | rf | RandomForest | 278 | 104 | 174 | 37.41% | 36.67% | 37.41% | 12.59 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 278 | 98 | 180 | 35.25% | 35.42% | 35.25% | 14.75 pp | -82 | 12 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 276 | 141 | 135 | 51.09% | 51.25% | 51.09% | 1.09 pp | 6 | 22 | 0.27 |
| BTC Market Hours | transformer | Transformer | 276 | 128 | 148 | 46.38% | 45.83% | 46.38% | 3.62 pp | -20 | 22 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 276 | 127 | 149 | 46.01% | 46.25% | 46.01% | 3.99 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 22 | -1.18 |
| BTC Market Hours | rf | RandomForest | 276 | 122 | 154 | 44.20% | 42.92% | 44.20% | 5.80 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 276 | 114 | 162 | 41.30% | 42.50% | 41.30% | 8.70 pp | -48 | 22 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 276 | 135 | 141 | 48.91% | 48.33% | 48.91% | 1.09 pp | -6 | 23 | -0.26 |
| BTC Market Hours Daily | transformer | Transformer | 276 | 134 | 142 | 48.55% | 47.92% | 48.55% | 1.45 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | nn | NN | 276 | 132 | 144 | 47.83% | 48.33% | 47.83% | 2.17 pp | -12 | 23 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 276 | 119 | 157 | 43.12% | 42.50% | 43.12% | 6.88 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | rf | RandomForest | 276 | 118 | 158 | 42.75% | 42.08% | 42.75% | 7.25 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 276 | 113 | 163 | 40.94% | 42.50% | 40.94% | 9.06 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
