# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T14:39:22.160709+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 324 | 264 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 360 | 300 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 537 | 288 | 249 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 537 | 288 | 249 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T01:00:00+00:00 | 252 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T01:00:00+00:00 | 252 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T01:00:00+00:00 | 252 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T01:00:00+00:00 | 253 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 288 | 142 | 146 | 49.31% | 49.17% | 49.31% | 0.69 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 288 | 140 | 148 | 48.61% | 50.42% | 48.61% | 1.39 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 288 | 138 | 150 | 47.92% | 47.92% | 47.92% | 2.08 pp | -12 | 24 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 264 | 127 | 137 | 48.11% | 47.50% | 48.11% | 1.89 pp | -10 | 11 | -0.91 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| Consolidated Hourly | rf | RandomForest | 252 | 117 | 135 | 46.43% | 46.25% | 46.43% | 3.57 pp | -18 | 16 | -1.12 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 252 | 117 | 135 | 46.43% | 46.25% | 46.43% | 3.57 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 8 | -1.38 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| Consolidated Hourly | lstm | LSTM | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 252 | 114 | 138 | 45.24% | 45.42% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 252 | 114 | 138 | 45.24% | 45.42% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| BTC Daily | nn | NN | 290 | 135 | 155 | 46.55% | 46.25% | 46.55% | 3.45 pp | -20 | 13 | -1.54 |
| BTC Market Hours Daily | xgb | XGBoost | 288 | 125 | 163 | 43.40% | 44.17% | 43.40% | 6.60 pp | -38 | 24 | -1.58 |
| BTC Hourly | transformer | Transformer | 264 | 123 | 141 | 46.59% | 47.08% | 46.59% | 3.41 pp | -18 | 11 | -1.64 |
| BTC Market Hours Daily | rf | RandomForest | 288 | 124 | 164 | 43.06% | 43.33% | 43.06% | 6.94 pp | -40 | 24 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 252 | 112 | 140 | 44.44% | 43.75% | 44.44% | 5.56 pp | -28 | 16 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 252 | 112 | 140 | 44.44% | 43.75% | 44.44% | 5.56 pp | -28 | 16 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 288 | 119 | 169 | 41.32% | 44.17% | 41.32% | 8.68 pp | -50 | 24 | -2.08 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 43.75% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
| Consolidated Hourly | nn | NN | 252 | 107 | 145 | 42.46% | 42.92% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Hourly | xgb | XGBoost | 252 | 107 | 145 | 42.46% | 42.08% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 252 | 107 | 145 | 42.46% | 42.92% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 252 | 107 | 145 | 42.46% | 42.08% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 264 | 110 | 154 | 41.67% | 40.83% | 41.67% | 8.33 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 290 | 117 | 173 | 40.34% | 37.92% | 40.34% | 9.66 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 264 | 106 | 158 | 40.15% | 40.42% | 40.15% | 9.85 pp | -52 | 11 | -4.73 |
| BTC Daily | rf | RandomForest | 290 | 111 | 179 | 38.28% | 37.08% | 38.28% | 11.72 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 300 | 113 | 187 | 37.67% | 37.92% | 37.67% | 12.33 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 290 | 103 | 187 | 35.52% | 35.83% | 35.52% | 14.48 pp | -84 | 13 | -6.46 |
| BTC Hourly | lstm | LSTM | 264 | 95 | 169 | 35.98% | 34.58% | 35.98% | 14.02 pp | -74 | 11 | -6.73 |
| BTC Hourly | xgb | XGBoost | 264 | 91 | 173 | 34.47% | 34.58% | 34.47% | 15.53 pp | -82 | 11 | -7.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 264 | 127 | 137 | 48.11% | 47.50% | 48.11% | 1.89 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 264 | 123 | 141 | 46.59% | 47.08% | 46.59% | 3.41 pp | -18 | 11 | -1.64 |
| BTC Hourly | nn | NN | 264 | 110 | 154 | 41.67% | 40.83% | 41.67% | 8.33 pp | -44 | 11 | -4.00 |
| BTC Hourly | rf | RandomForest | 264 | 106 | 158 | 40.15% | 40.42% | 40.15% | 9.85 pp | -52 | 11 | -4.73 |
| BTC Hourly | lstm | LSTM | 264 | 95 | 169 | 35.98% | 34.58% | 35.98% | 14.02 pp | -74 | 11 | -6.73 |
| BTC Hourly | xgb | XGBoost | 264 | 91 | 173 | 34.47% | 34.58% | 34.47% | 15.53 pp | -82 | 11 | -7.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 290 | 135 | 155 | 46.55% | 46.25% | 46.55% | 3.45 pp | -20 | 13 | -1.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 43.75% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
| BTC Daily | transformer | Transformer | 290 | 117 | 173 | 40.34% | 37.92% | 40.34% | 9.66 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 290 | 111 | 179 | 38.28% | 37.08% | 38.28% | 11.72 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 300 | 113 | 187 | 37.67% | 37.92% | 37.67% | 12.33 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 290 | 103 | 187 | 35.52% | 35.83% | 35.52% | 14.48 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 288 | 142 | 146 | 49.31% | 49.17% | 49.31% | 0.69 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 288 | 140 | 148 | 48.61% | 50.42% | 48.61% | 1.39 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 288 | 138 | 150 | 47.92% | 47.92% | 47.92% | 2.08 pp | -12 | 24 | -0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 288 | 125 | 163 | 43.40% | 44.17% | 43.40% | 6.60 pp | -38 | 24 | -1.58 |
| BTC Market Hours Daily | rf | RandomForest | 288 | 124 | 164 | 43.06% | 43.33% | 43.06% | 6.94 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 288 | 119 | 169 | 41.32% | 44.17% | 41.32% | 8.68 pp | -50 | 24 | -2.08 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 252 | 117 | 135 | 46.43% | 46.25% | 46.43% | 3.57 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | lstm | LSTM | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 252 | 114 | 138 | 45.24% | 45.42% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 252 | 112 | 140 | 44.44% | 43.75% | 44.44% | 5.56 pp | -28 | 16 | -1.75 |
| Consolidated Hourly | nn | NN | 252 | 107 | 145 | 42.46% | 42.92% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Hourly | xgb | XGBoost | 252 | 107 | 145 | 42.46% | 42.08% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 252 | 117 | 135 | 46.43% | 46.25% | 46.43% | 3.57 pp | -18 | 16 | -1.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 252 | 114 | 138 | 45.24% | 44.17% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 252 | 114 | 138 | 45.24% | 45.42% | 45.24% | 4.76 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 252 | 112 | 140 | 44.44% | 43.75% | 44.44% | 5.56 pp | -28 | 16 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 252 | 107 | 145 | 42.46% | 42.92% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 252 | 107 | 145 | 42.46% | 42.08% | 42.46% | 7.54 pp | -38 | 16 | -2.38 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
