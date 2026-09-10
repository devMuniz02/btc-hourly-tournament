# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T21:23:09.841876+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 349 | 289 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 520 | 277 | 243 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 519 | 276 | 243 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 277 | 141 | 136 | 50.90% | 50.83% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 276 | 135 | 141 | 48.91% | 48.33% | 48.91% | 1.09 pp | -6 | 23 | -0.26 |
| BTC Market Hours Daily | transformer | Transformer | 276 | 134 | 142 | 48.55% | 47.92% | 48.55% | 1.45 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | nn | NN | 276 | 132 | 144 | 47.83% | 48.33% | 47.83% | 2.17 pp | -12 | 23 | -0.52 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 253 | 123 | 130 | 48.62% | 48.33% | 48.62% | 1.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 277 | 128 | 149 | 46.21% | 46.67% | 46.21% | 3.79 pp | -21 | 22 | -0.95 |
| BTC Market Hours | transformer | Transformer | 277 | 128 | 149 | 46.21% | 45.83% | 46.21% | 3.79 pp | -21 | 22 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| BTC Market Hours | xgb | XGBoost | 277 | 125 | 152 | 45.13% | 45.00% | 45.13% | 4.87 pp | -27 | 22 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 277 | 122 | 155 | 44.04% | 42.92% | 44.04% | 5.96 pp | -33 | 22 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 276 | 119 | 157 | 43.12% | 42.50% | 43.12% | 6.88 pp | -38 | 23 | -1.65 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 276 | 118 | 158 | 42.75% | 42.08% | 42.75% | 7.25 pp | -40 | 23 | -1.74 |
| BTC Daily | nn | NN | 279 | 128 | 151 | 45.88% | 45.00% | 45.88% | 4.12 pp | -23 | 12 | -1.92 |
| BTC Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 46.25% | 45.45% | 4.55 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 276 | 113 | 163 | 40.94% | 42.50% | 40.94% | 9.06 pp | -50 | 23 | -2.17 |
| BTC Market Hours | lstm | LSTM | 277 | 114 | 163 | 41.16% | 42.50% | 41.16% | 8.84 pp | -49 | 22 | -2.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 279 | 126 | 153 | 45.16% | 44.58% | 45.16% | 4.84 pp | -27 | 12 | -2.25 |
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
| BTC Daily | transformer | Transformer | 279 | 113 | 166 | 40.50% | 37.92% | 40.50% | 9.50 pp | -53 | 12 | -4.42 |
| BTC Daily | xgb | XGBoost | 289 | 108 | 181 | 37.37% | 37.50% | 37.37% | 12.63 pp | -73 | 13 | -5.62 |
| BTC Daily | rf | RandomForest | 279 | 105 | 174 | 37.63% | 36.67% | 37.63% | 12.37 pp | -69 | 12 | -5.75 |
| BTC Hourly | lstm | LSTM | 253 | 93 | 160 | 36.76% | 36.25% | 36.76% | 13.24 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 279 | 98 | 181 | 35.13% | 35.42% | 35.13% | 14.87 pp | -83 | 12 | -6.92 |
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
| BTC Daily | nn | NN | 279 | 128 | 151 | 45.88% | 45.00% | 45.88% | 4.12 pp | -23 | 12 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 279 | 126 | 153 | 45.16% | 44.58% | 45.16% | 4.84 pp | -27 | 12 | -2.25 |
| BTC Daily | transformer | Transformer | 279 | 113 | 166 | 40.50% | 37.92% | 40.50% | 9.50 pp | -53 | 12 | -4.42 |
| BTC Daily | xgb | XGBoost | 289 | 108 | 181 | 37.37% | 37.50% | 37.37% | 12.63 pp | -73 | 13 | -5.62 |
| BTC Daily | rf | RandomForest | 279 | 105 | 174 | 37.63% | 36.67% | 37.63% | 12.37 pp | -69 | 12 | -5.75 |
| BTC Daily | lstm | LSTM | 279 | 98 | 181 | 35.13% | 35.42% | 35.13% | 14.87 pp | -83 | 12 | -6.92 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 277 | 141 | 136 | 50.90% | 50.83% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 277 | 128 | 149 | 46.21% | 46.67% | 46.21% | 3.79 pp | -21 | 22 | -0.95 |
| BTC Market Hours | transformer | Transformer | 277 | 128 | 149 | 46.21% | 45.83% | 46.21% | 3.79 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 277 | 125 | 152 | 45.13% | 45.00% | 45.13% | 4.87 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 277 | 122 | 155 | 44.04% | 42.92% | 44.04% | 5.96 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 277 | 114 | 163 | 41.16% | 42.50% | 41.16% | 8.84 pp | -49 | 22 | -2.23 |

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
