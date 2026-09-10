# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T10:44:22.146350+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 306 | 246 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 341 | 281 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 503 | 269 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 503 | 269 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 269 | 141 | 128 | 52.42% | 51.67% | 52.42% | 2.42 pp | 13 | 21 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 269 | 133 | 136 | 49.44% | 48.33% | 49.44% | 0.56 pp | -3 | 22 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 269 | 131 | 138 | 48.70% | 48.33% | 48.70% | 1.30 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | nn | NN | 269 | 130 | 139 | 48.33% | 48.75% | 48.33% | 1.67 pp | -9 | 22 | -0.41 |
| Consolidated Hourly | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 246 | 119 | 127 | 48.37% | 48.75% | 48.37% | 1.63 pp | -8 | 11 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 269 | 125 | 144 | 46.47% | 46.25% | 46.47% | 3.53 pp | -19 | 21 | -0.90 |
| BTC Market Hours | transformer | Transformer | 269 | 125 | 144 | 46.47% | 46.25% | 46.47% | 3.53 pp | -19 | 21 | -0.90 |
| BTC Market Hours | xgb | XGBoost | 269 | 124 | 145 | 46.10% | 45.00% | 46.10% | 3.90 pp | -21 | 21 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 269 | 119 | 150 | 44.24% | 42.92% | 44.24% | 5.76 pp | -31 | 22 | -1.41 |
| BTC Market Hours | rf | RandomForest | 269 | 119 | 150 | 44.24% | 42.50% | 44.24% | 5.76 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 269 | 115 | 154 | 42.75% | 41.25% | 42.75% | 7.25 pp | -39 | 22 | -1.77 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 271 | 124 | 147 | 45.76% | 45.00% | 45.76% | 4.24 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 271 | 124 | 147 | 45.76% | 45.42% | 45.76% | 4.24 pp | -23 | 12 | -1.92 |
| BTC Hourly | transformer | Transformer | 246 | 111 | 135 | 45.12% | 45.83% | 45.12% | 4.88 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 269 | 110 | 159 | 40.89% | 42.08% | 40.89% | 9.11 pp | -49 | 22 | -2.23 |
| BTC Market Hours | lstm | LSTM | 269 | 110 | 159 | 40.89% | 42.08% | 40.89% | 9.11 pp | -49 | 21 | -2.33 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 246 | 104 | 142 | 42.28% | 42.08% | 42.28% | 7.72 pp | -38 | 11 | -3.45 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 246 | 101 | 145 | 41.06% | 41.67% | 41.06% | 8.94 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 271 | 109 | 162 | 40.22% | 37.92% | 40.22% | 9.78 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 271 | 103 | 168 | 38.01% | 37.50% | 38.01% | 11.99 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 281 | 104 | 177 | 37.01% | 37.08% | 37.01% | 12.99 pp | -73 | 13 | -5.62 |
| BTC Hourly | lstm | LSTM | 246 | 91 | 155 | 36.99% | 37.08% | 36.99% | 13.01 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 271 | 96 | 175 | 35.42% | 35.00% | 35.42% | 14.58 pp | -79 | 12 | -6.58 |
| BTC Hourly | xgb | XGBoost | 246 | 86 | 160 | 34.96% | 35.42% | 34.96% | 15.04 pp | -74 | 11 | -6.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 246 | 119 | 127 | 48.37% | 48.75% | 48.37% | 1.63 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 246 | 111 | 135 | 45.12% | 45.83% | 45.12% | 4.88 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 246 | 104 | 142 | 42.28% | 42.08% | 42.28% | 7.72 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 246 | 101 | 145 | 41.06% | 41.67% | 41.06% | 8.94 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 246 | 91 | 155 | 36.99% | 37.08% | 36.99% | 13.01 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 246 | 86 | 160 | 34.96% | 35.42% | 34.96% | 15.04 pp | -74 | 11 | -6.73 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 271 | 124 | 147 | 45.76% | 45.00% | 45.76% | 4.24 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 271 | 124 | 147 | 45.76% | 45.42% | 45.76% | 4.24 pp | -23 | 12 | -1.92 |
| BTC Daily | transformer | Transformer | 271 | 109 | 162 | 40.22% | 37.92% | 40.22% | 9.78 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 271 | 103 | 168 | 38.01% | 37.50% | 38.01% | 11.99 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 281 | 104 | 177 | 37.01% | 37.08% | 37.01% | 12.99 pp | -73 | 13 | -5.62 |
| BTC Daily | lstm | LSTM | 271 | 96 | 175 | 35.42% | 35.00% | 35.42% | 14.58 pp | -79 | 12 | -6.58 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 269 | 141 | 128 | 52.42% | 51.67% | 52.42% | 2.42 pp | 13 | 21 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 269 | 125 | 144 | 46.47% | 46.25% | 46.47% | 3.53 pp | -19 | 21 | -0.90 |
| BTC Market Hours | transformer | Transformer | 269 | 125 | 144 | 46.47% | 46.25% | 46.47% | 3.53 pp | -19 | 21 | -0.90 |
| BTC Market Hours | xgb | XGBoost | 269 | 124 | 145 | 46.10% | 45.00% | 46.10% | 3.90 pp | -21 | 21 | -1.00 |
| BTC Market Hours | rf | RandomForest | 269 | 119 | 150 | 44.24% | 42.50% | 44.24% | 5.76 pp | -31 | 21 | -1.48 |
| BTC Market Hours | lstm | LSTM | 269 | 110 | 159 | 40.89% | 42.08% | 40.89% | 9.11 pp | -49 | 21 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 269 | 133 | 136 | 49.44% | 48.33% | 49.44% | 0.56 pp | -3 | 22 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 269 | 131 | 138 | 48.70% | 48.33% | 48.70% | 1.30 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | nn | NN | 269 | 130 | 139 | 48.33% | 48.75% | 48.33% | 1.67 pp | -9 | 22 | -0.41 |
| BTC Market Hours Daily | xgb | XGBoost | 269 | 119 | 150 | 44.24% | 42.92% | 44.24% | 5.76 pp | -31 | 22 | -1.41 |
| BTC Market Hours Daily | rf | RandomForest | 269 | 115 | 154 | 42.75% | 41.25% | 42.75% | 7.25 pp | -39 | 22 | -1.77 |
| BTC Market Hours Daily | lstm | LSTM | 269 | 110 | 159 | 40.89% | 42.08% | 40.89% | 9.11 pp | -49 | 22 | -2.23 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Hourly | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
