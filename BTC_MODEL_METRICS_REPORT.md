# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T16:13:12.263933+00:00
Scope: `all`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1279 | 991 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1154 | 789 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 15:00:00+00:00 | 865 | 551 | 313 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 15:00:00+00:00 | 867 | 605 | 260 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 551 | 267 | 284 | 48.46% | 46.25% | 47.50% | 1.54 pp | -17 | 52 | -0.33 |
| BTC Market Hours | nn | NN | 551 | 263 | 288 | 47.73% | 51.67% | 49.58% | 2.27 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 551 | 261 | 290 | 47.37% | 47.50% | 48.12% | 2.63 pp | -29 | 52 | -0.56 |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 605 | 284 | 321 | 46.94% | 49.58% | 47.92% | 3.06 pp | -37 | 52 | -0.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 779 | 373 | 406 | 47.88% | 45.00% | 47.08% | 2.12 pp | -33 | 45 | -0.73 |
| BTC Market Hours Daily | nn | NN | 605 | 281 | 324 | 46.45% | 47.08% | 47.92% | 3.55 pp | -43 | 52 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 957 | 456 | 501 | 47.65% | 49.58% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 605 | 279 | 326 | 46.12% | 48.33% | 46.88% | 3.88 pp | -47 | 52 | -0.90 |
| BTC Daily | transformer | Transformer | 779 | 361 | 418 | 46.34% | 40.83% | 46.46% | 3.66 pp | -57 | 45 | -1.27 |
| BTC Hourly | transformer | Transformer | 957 | 446 | 511 | 46.60% | 45.00% | 43.96% | 3.40 pp | -65 | 50 | -1.30 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| BTC Daily | nn | NN | 779 | 360 | 419 | 46.21% | 44.17% | 45.00% | 3.79 pp | -59 | 45 | -1.31 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 551 | 238 | 313 | 43.19% | 45.00% | 42.92% | 6.81 pp | -75 | 52 | -1.44 |
| BTC Market Hours | lstm | LSTM | 551 | 237 | 314 | 43.01% | 41.25% | 43.12% | 6.99 pp | -77 | 52 | -1.48 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 551 | 231 | 320 | 41.92% | 44.17% | 41.88% | 8.08 pp | -89 | 52 | -1.71 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 605 | 253 | 352 | 41.82% | 43.75% | 41.04% | 8.18 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 957 | 425 | 532 | 44.41% | 43.75% | 43.54% | 5.59 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 957 | 424 | 533 | 44.31% | 42.92% | 42.92% | 5.69 pp | -109 | 50 | -2.18 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 605 | 244 | 361 | 40.33% | 39.58% | 40.42% | 9.67 pp | -117 | 52 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 605 | 243 | 362 | 40.17% | 40.83% | 39.38% | 9.83 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 779 | 327 | 452 | 41.98% | 34.17% | 39.58% | 8.02 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 957 | 409 | 548 | 42.74% | 37.50% | 42.29% | 7.26 pp | -139 | 50 | -2.78 |
| BTC Daily | rf | RandomForest | 779 | 326 | 453 | 41.85% | 38.75% | 41.88% | 8.15 pp | -127 | 45 | -2.82 |
| BTC Hourly | xgb | XGBoost | 957 | 397 | 560 | 41.48% | 38.75% | 39.58% | 8.52 pp | -163 | 50 | -3.26 |
| BTC Daily | xgb | XGBoost | 789 | 306 | 483 | 38.78% | 34.58% | 35.83% | 11.22 pp | -177 | 45 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 957 | 456 | 501 | 47.65% | 49.58% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 957 | 446 | 511 | 46.60% | 45.00% | 43.96% | 3.40 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 957 | 425 | 532 | 44.41% | 43.75% | 43.54% | 5.59 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 957 | 424 | 533 | 44.31% | 42.92% | 42.92% | 5.69 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 957 | 409 | 548 | 42.74% | 37.50% | 42.29% | 7.26 pp | -139 | 50 | -2.78 |
| BTC Hourly | xgb | XGBoost | 957 | 397 | 560 | 41.48% | 38.75% | 39.58% | 8.52 pp | -163 | 50 | -3.26 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 779 | 373 | 406 | 47.88% | 45.00% | 47.08% | 2.12 pp | -33 | 45 | -0.73 |
| BTC Daily | transformer | Transformer | 779 | 361 | 418 | 46.34% | 40.83% | 46.46% | 3.66 pp | -57 | 45 | -1.27 |
| BTC Daily | nn | NN | 779 | 360 | 419 | 46.21% | 44.17% | 45.00% | 3.79 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 779 | 327 | 452 | 41.98% | 34.17% | 39.58% | 8.02 pp | -125 | 45 | -2.78 |
| BTC Daily | rf | RandomForest | 779 | 326 | 453 | 41.85% | 38.75% | 41.88% | 8.15 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 789 | 306 | 483 | 38.78% | 34.58% | 35.83% | 11.22 pp | -177 | 45 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 551 | 267 | 284 | 48.46% | 46.25% | 47.50% | 1.54 pp | -17 | 52 | -0.33 |
| BTC Market Hours | nn | NN | 551 | 263 | 288 | 47.73% | 51.67% | 49.58% | 2.27 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 551 | 261 | 290 | 47.37% | 47.50% | 48.12% | 2.63 pp | -29 | 52 | -0.56 |
| BTC Market Hours | rf | RandomForest | 551 | 238 | 313 | 43.19% | 45.00% | 42.92% | 6.81 pp | -75 | 52 | -1.44 |
| BTC Market Hours | lstm | LSTM | 551 | 237 | 314 | 43.01% | 41.25% | 43.12% | 6.99 pp | -77 | 52 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 551 | 231 | 320 | 41.92% | 44.17% | 41.88% | 8.08 pp | -89 | 52 | -1.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 605 | 284 | 321 | 46.94% | 49.58% | 47.92% | 3.06 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 605 | 281 | 324 | 46.45% | 47.08% | 47.92% | 3.55 pp | -43 | 52 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 605 | 279 | 326 | 46.12% | 48.33% | 46.88% | 3.88 pp | -47 | 52 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 605 | 253 | 352 | 41.82% | 43.75% | 41.04% | 8.18 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 605 | 244 | 361 | 40.33% | 39.58% | 40.42% | 9.67 pp | -117 | 52 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 605 | 243 | 362 | 40.17% | 40.83% | 39.38% | 9.83 pp | -119 | 52 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
