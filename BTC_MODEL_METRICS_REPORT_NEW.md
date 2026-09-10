# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T22:59:36.099950+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 314 | 254 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 350 | 290 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 21:00:00+00:00 | 522 | 278 | 244 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 21:00:00+00:00 | 521 | 277 | 244 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 278 | 141 | 137 | 50.72% | 50.42% | 50.72% | 0.72 pp | 4 | 22 | 0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 277 | 136 | 141 | 49.10% | 48.75% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 277 | 134 | 143 | 48.38% | 47.92% | 48.38% | 1.62 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | nn | NN | 277 | 133 | 144 | 48.01% | 48.75% | 48.01% | 1.99 pp | -11 | 23 | -0.48 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 254 | 123 | 131 | 48.43% | 48.33% | 48.43% | 1.57 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 278 | 128 | 150 | 46.04% | 46.25% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 278 | 128 | 150 | 46.04% | 45.83% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| BTC Market Hours | xgb | XGBoost | 278 | 125 | 153 | 44.96% | 45.00% | 44.96% | 5.04 pp | -28 | 22 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 278 | 122 | 156 | 43.88% | 42.92% | 43.88% | 6.12 pp | -34 | 22 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 277 | 120 | 157 | 43.32% | 42.92% | 43.32% | 6.68 pp | -37 | 23 | -1.61 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 277 | 119 | 158 | 42.96% | 42.50% | 42.96% | 7.04 pp | -39 | 23 | -1.70 |
| BTC Daily | nn | NN | 280 | 128 | 152 | 45.71% | 45.00% | 45.71% | 4.29 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 277 | 114 | 163 | 41.16% | 42.92% | 41.16% | 8.84 pp | -49 | 23 | -2.13 |
| BTC Daily | mlp_sklearn | MLPClassifier | 280 | 127 | 153 | 45.36% | 44.58% | 45.36% | 4.64 pp | -26 | 12 | -2.17 |
| BTC Hourly | transformer | Transformer | 254 | 115 | 139 | 45.28% | 45.83% | 45.28% | 4.72 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 278 | 114 | 164 | 41.01% | 42.50% | 41.01% | 8.99 pp | -50 | 22 | -2.27 |
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
| BTC Hourly | nn | NN | 254 | 107 | 147 | 42.13% | 42.50% | 42.13% | 7.87 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 254 | 105 | 149 | 41.34% | 42.08% | 41.34% | 8.66 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 280 | 113 | 167 | 40.36% | 37.50% | 40.36% | 9.64 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 280 | 106 | 174 | 37.86% | 36.67% | 37.86% | 12.14 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 290 | 108 | 182 | 37.24% | 37.50% | 37.24% | 12.76 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 254 | 94 | 160 | 37.01% | 36.67% | 37.01% | 12.99 pp | -66 | 11 | -6.00 |
| BTC Daily | lstm | LSTM | 280 | 99 | 181 | 35.36% | 35.42% | 35.36% | 14.64 pp | -82 | 12 | -6.83 |
| BTC Hourly | xgb | XGBoost | 254 | 89 | 165 | 35.04% | 35.83% | 35.04% | 14.96 pp | -76 | 11 | -6.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 254 | 123 | 131 | 48.43% | 48.33% | 48.43% | 1.57 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 254 | 115 | 139 | 45.28% | 45.83% | 45.28% | 4.72 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 254 | 107 | 147 | 42.13% | 42.50% | 42.13% | 7.87 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 254 | 105 | 149 | 41.34% | 42.08% | 41.34% | 8.66 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 254 | 94 | 160 | 37.01% | 36.67% | 37.01% | 12.99 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 254 | 89 | 165 | 35.04% | 35.83% | 35.04% | 14.96 pp | -76 | 11 | -6.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 280 | 128 | 152 | 45.71% | 45.00% | 45.71% | 4.29 pp | -24 | 12 | -2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 280 | 127 | 153 | 45.36% | 44.58% | 45.36% | 4.64 pp | -26 | 12 | -2.17 |
| BTC Daily | transformer | Transformer | 280 | 113 | 167 | 40.36% | 37.50% | 40.36% | 9.64 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 280 | 106 | 174 | 37.86% | 36.67% | 37.86% | 12.14 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 290 | 108 | 182 | 37.24% | 37.50% | 37.24% | 12.76 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 280 | 99 | 181 | 35.36% | 35.42% | 35.36% | 14.64 pp | -82 | 12 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 278 | 141 | 137 | 50.72% | 50.42% | 50.72% | 0.72 pp | 4 | 22 | 0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 278 | 128 | 150 | 46.04% | 46.25% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 278 | 128 | 150 | 46.04% | 45.83% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 278 | 125 | 153 | 44.96% | 45.00% | 44.96% | 5.04 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 278 | 122 | 156 | 43.88% | 42.92% | 43.88% | 6.12 pp | -34 | 22 | -1.55 |
| BTC Market Hours | lstm | LSTM | 278 | 114 | 164 | 41.01% | 42.50% | 41.01% | 8.99 pp | -50 | 22 | -2.27 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 277 | 136 | 141 | 49.10% | 48.75% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 277 | 134 | 143 | 48.38% | 47.92% | 48.38% | 1.62 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | nn | NN | 277 | 133 | 144 | 48.01% | 48.75% | 48.01% | 1.99 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | xgb | XGBoost | 277 | 120 | 157 | 43.32% | 42.92% | 43.32% | 6.68 pp | -37 | 23 | -1.61 |
| BTC Market Hours Daily | rf | RandomForest | 277 | 119 | 158 | 42.96% | 42.50% | 42.96% | 7.04 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 277 | 114 | 163 | 41.16% | 42.92% | 41.16% | 8.84 pp | -49 | 23 | -2.13 |

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
