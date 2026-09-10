# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T09:27:49.009270+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 305 | 245 | 60 | 0 |
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
| BTC Hourly | mlp_sklearn | MLPClassifier | 245 | 118 | 127 | 48.16% | 48.33% | 48.16% | 1.84 pp | -9 | 11 | -0.82 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 271 | 125 | 146 | 46.13% | 45.00% | 46.13% | 3.87 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 271 | 125 | 146 | 46.13% | 45.42% | 46.13% | 3.87 pp | -21 | 12 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 269 | 115 | 154 | 42.75% | 41.25% | 42.75% | 7.25 pp | -39 | 22 | -1.77 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Hourly | transformer | Transformer | 245 | 111 | 134 | 45.31% | 46.25% | 45.31% | 4.69 pp | -23 | 11 | -2.09 |
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
| BTC Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.08% | 42.04% | 7.96 pp | -39 | 11 | -3.55 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 245 | 100 | 145 | 40.82% | 41.67% | 40.82% | 9.18 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 271 | 110 | 161 | 40.59% | 38.33% | 40.59% | 9.41 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 271 | 104 | 167 | 38.38% | 37.50% | 38.38% | 11.62 pp | -63 | 12 | -5.25 |
| BTC Daily | xgb | XGBoost | 281 | 105 | 176 | 37.37% | 37.50% | 37.37% | 12.63 pp | -71 | 13 | -5.46 |
| BTC Hourly | lstm | LSTM | 245 | 90 | 155 | 36.73% | 37.08% | 36.73% | 13.27 pp | -65 | 11 | -5.91 |
| BTC Daily | lstm | LSTM | 271 | 96 | 175 | 35.42% | 35.42% | 35.42% | 14.58 pp | -79 | 12 | -6.58 |
| BTC Hourly | xgb | XGBoost | 245 | 86 | 159 | 35.10% | 35.83% | 35.10% | 14.90 pp | -73 | 11 | -6.64 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 245 | 118 | 127 | 48.16% | 48.33% | 48.16% | 1.84 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 245 | 111 | 134 | 45.31% | 46.25% | 45.31% | 4.69 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.08% | 42.04% | 7.96 pp | -39 | 11 | -3.55 |
| BTC Hourly | rf | RandomForest | 245 | 100 | 145 | 40.82% | 41.67% | 40.82% | 9.18 pp | -45 | 11 | -4.09 |
| BTC Hourly | lstm | LSTM | 245 | 90 | 155 | 36.73% | 37.08% | 36.73% | 13.27 pp | -65 | 11 | -5.91 |
| BTC Hourly | xgb | XGBoost | 245 | 86 | 159 | 35.10% | 35.83% | 35.10% | 14.90 pp | -73 | 11 | -6.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 271 | 125 | 146 | 46.13% | 45.00% | 46.13% | 3.87 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 271 | 125 | 146 | 46.13% | 45.42% | 46.13% | 3.87 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 271 | 110 | 161 | 40.59% | 38.33% | 40.59% | 9.41 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 271 | 104 | 167 | 38.38% | 37.50% | 38.38% | 11.62 pp | -63 | 12 | -5.25 |
| BTC Daily | xgb | XGBoost | 281 | 105 | 176 | 37.37% | 37.50% | 37.37% | 12.63 pp | -71 | 13 | -5.46 |
| BTC Daily | lstm | LSTM | 271 | 96 | 175 | 35.42% | 35.42% | 35.42% | 14.58 pp | -79 | 12 | -6.58 |

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
