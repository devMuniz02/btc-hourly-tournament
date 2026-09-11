# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T00:21:19.316340+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 315 | 255 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 350 | 290 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 23:00:00+00:00 | 524 | 278 | 246 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 23:00:00+00:00 | 524 | 278 | 246 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 278 | 141 | 137 | 50.72% | 50.42% | 50.72% | 0.72 pp | 4 | 22 | 0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 278 | 137 | 141 | 49.28% | 48.75% | 49.28% | 0.72 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 278 | 134 | 144 | 48.20% | 47.92% | 48.20% | 1.80 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | nn | NN | 278 | 133 | 145 | 47.84% | 48.33% | 47.84% | 2.16 pp | -12 | 23 | -0.52 |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 255 | 123 | 132 | 48.24% | 48.33% | 48.24% | 1.76 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 278 | 128 | 150 | 46.04% | 46.25% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 278 | 128 | 150 | 46.04% | 45.83% | 46.04% | 3.96 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| BTC Market Hours | xgb | XGBoost | 278 | 125 | 153 | 44.96% | 45.00% | 44.96% | 5.04 pp | -28 | 22 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 278 | 122 | 156 | 43.88% | 42.92% | 43.88% | 6.12 pp | -34 | 22 | -1.55 |
| BTC Market Hours Daily | xgb | XGBoost | 278 | 120 | 158 | 43.17% | 42.92% | 43.17% | 6.83 pp | -38 | 23 | -1.65 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 278 | 119 | 159 | 42.81% | 42.50% | 42.81% | 7.19 pp | -40 | 23 | -1.74 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| BTC Daily | nn | NN | 280 | 128 | 152 | 45.71% | 45.00% | 45.71% | 4.29 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 280 | 126 | 154 | 45.00% | 44.58% | 45.00% | 5.00 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 278 | 114 | 164 | 41.01% | 42.92% | 41.01% | 8.99 pp | -50 | 23 | -2.17 |
| BTC Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 45.83% | 45.10% | 4.90 pp | -25 | 11 | -2.27 |
| BTC Market Hours | lstm | LSTM | 278 | 114 | 164 | 41.01% | 42.50% | 41.01% | 8.99 pp | -50 | 22 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 255 | 107 | 148 | 41.96% | 42.50% | 41.96% | 8.04 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 280 | 112 | 168 | 40.00% | 37.50% | 40.00% | 10.00 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 280 | 105 | 175 | 37.50% | 36.67% | 37.50% | 12.50 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 290 | 107 | 183 | 36.90% | 37.08% | 36.90% | 13.10 pp | -76 | 14 | -5.43 |
| BTC Hourly | lstm | LSTM | 255 | 94 | 161 | 36.86% | 36.25% | 36.86% | 13.14 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 280 | 100 | 180 | 35.71% | 35.83% | 35.71% | 14.29 pp | -80 | 13 | -6.15 |
| BTC Hourly | xgb | XGBoost | 255 | 89 | 166 | 34.90% | 35.42% | 34.90% | 15.10 pp | -77 | 11 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 255 | 123 | 132 | 48.24% | 48.33% | 48.24% | 1.76 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 45.83% | 45.10% | 4.90 pp | -25 | 11 | -2.27 |
| BTC Hourly | nn | NN | 255 | 107 | 148 | 41.96% | 42.50% | 41.96% | 8.04 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 11 | -4.09 |
| BTC Hourly | lstm | LSTM | 255 | 94 | 161 | 36.86% | 36.25% | 36.86% | 13.14 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 255 | 89 | 166 | 34.90% | 35.42% | 34.90% | 15.10 pp | -77 | 11 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 280 | 128 | 152 | 45.71% | 45.00% | 45.71% | 4.29 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 280 | 126 | 154 | 45.00% | 44.58% | 45.00% | 5.00 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 280 | 112 | 168 | 40.00% | 37.50% | 40.00% | 10.00 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 280 | 105 | 175 | 37.50% | 36.67% | 37.50% | 12.50 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 290 | 107 | 183 | 36.90% | 37.08% | 36.90% | 13.10 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 280 | 100 | 180 | 35.71% | 35.83% | 35.71% | 14.29 pp | -80 | 13 | -6.15 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 278 | 137 | 141 | 49.28% | 48.75% | 49.28% | 0.72 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 278 | 134 | 144 | 48.20% | 47.92% | 48.20% | 1.80 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | nn | NN | 278 | 133 | 145 | 47.84% | 48.33% | 47.84% | 2.16 pp | -12 | 23 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 278 | 120 | 158 | 43.17% | 42.92% | 43.17% | 6.83 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | rf | RandomForest | 278 | 119 | 159 | 42.81% | 42.50% | 42.81% | 7.19 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 278 | 114 | 164 | 41.01% | 42.92% | 41.01% | 8.99 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
