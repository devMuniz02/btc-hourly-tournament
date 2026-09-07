# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T14:54:00.427104+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1278 | 990 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1154 | 789 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 13:00:00+00:00 | 863 | 551 | 311 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 13:00:00+00:00 | 864 | 604 | 258 | 2 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 779 | 374 | 405 | 48.01% | 45.42% | 47.29% | 1.99 pp | -31 | 45 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 604 | 284 | 320 | 47.02% | 50.00% | 47.92% | 2.98 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 604 | 280 | 324 | 46.36% | 46.67% | 47.71% | 3.64 pp | -44 | 52 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 956 | 456 | 500 | 47.70% | 49.58% | 47.29% | 2.30 pp | -44 | 50 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 604 | 279 | 325 | 46.19% | 48.75% | 47.08% | 3.81 pp | -46 | 52 | -0.88 |
| BTC Daily | transformer | Transformer | 779 | 362 | 417 | 46.47% | 41.25% | 46.67% | 3.53 pp | -55 | 45 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| BTC Daily | nn | NN | 779 | 360 | 419 | 46.21% | 44.17% | 45.00% | 3.79 pp | -59 | 45 | -1.31 |
| BTC Hourly | transformer | Transformer | 956 | 445 | 511 | 46.55% | 44.58% | 43.96% | 3.45 pp | -66 | 50 | -1.32 |
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
| BTC Market Hours Daily | rf | RandomForest | 604 | 253 | 351 | 41.89% | 44.17% | 41.25% | 8.11 pp | -98 | 52 | -1.88 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 956 | 424 | 532 | 44.35% | 43.33% | 43.33% | 5.65 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 956 | 423 | 533 | 44.25% | 42.50% | 42.71% | 5.75 pp | -110 | 50 | -2.20 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 604 | 243 | 361 | 40.23% | 39.17% | 40.21% | 9.77 pp | -118 | 52 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 41.25% | 39.58% | 9.77 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Hourly | lstm | LSTM | 956 | 409 | 547 | 42.78% | 37.50% | 42.29% | 7.22 pp | -138 | 50 | -2.76 |
| BTC Daily | lstm | LSTM | 779 | 327 | 452 | 41.98% | 34.17% | 39.58% | 8.02 pp | -125 | 45 | -2.78 |
| BTC Daily | rf | RandomForest | 779 | 326 | 453 | 41.85% | 38.75% | 41.88% | 8.15 pp | -127 | 45 | -2.82 |
| BTC Hourly | xgb | XGBoost | 956 | 397 | 559 | 41.53% | 38.75% | 39.79% | 8.47 pp | -162 | 50 | -3.24 |
| BTC Daily | xgb | XGBoost | 789 | 307 | 482 | 38.91% | 35.00% | 36.04% | 11.09 pp | -175 | 45 | -3.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 956 | 456 | 500 | 47.70% | 49.58% | 47.29% | 2.30 pp | -44 | 50 | -0.88 |
| BTC Hourly | transformer | Transformer | 956 | 445 | 511 | 46.55% | 44.58% | 43.96% | 3.45 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 956 | 424 | 532 | 44.35% | 43.33% | 43.33% | 5.65 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 956 | 423 | 533 | 44.25% | 42.50% | 42.71% | 5.75 pp | -110 | 50 | -2.20 |
| BTC Hourly | lstm | LSTM | 956 | 409 | 547 | 42.78% | 37.50% | 42.29% | 7.22 pp | -138 | 50 | -2.76 |
| BTC Hourly | xgb | XGBoost | 956 | 397 | 559 | 41.53% | 38.75% | 39.79% | 8.47 pp | -162 | 50 | -3.24 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 779 | 374 | 405 | 48.01% | 45.42% | 47.29% | 1.99 pp | -31 | 45 | -0.69 |
| BTC Daily | transformer | Transformer | 779 | 362 | 417 | 46.47% | 41.25% | 46.67% | 3.53 pp | -55 | 45 | -1.22 |
| BTC Daily | nn | NN | 779 | 360 | 419 | 46.21% | 44.17% | 45.00% | 3.79 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 779 | 327 | 452 | 41.98% | 34.17% | 39.58% | 8.02 pp | -125 | 45 | -2.78 |
| BTC Daily | rf | RandomForest | 779 | 326 | 453 | 41.85% | 38.75% | 41.88% | 8.15 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 789 | 307 | 482 | 38.91% | 35.00% | 36.04% | 11.09 pp | -175 | 45 | -3.89 |

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
| BTC Market Hours Daily | transformer | Transformer | 604 | 284 | 320 | 47.02% | 50.00% | 47.92% | 2.98 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 604 | 280 | 324 | 46.36% | 46.67% | 47.71% | 3.64 pp | -44 | 52 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 604 | 279 | 325 | 46.19% | 48.75% | 47.08% | 3.81 pp | -46 | 52 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 604 | 253 | 351 | 41.89% | 44.17% | 41.25% | 8.11 pp | -98 | 52 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 604 | 243 | 361 | 40.23% | 39.17% | 40.21% | 9.77 pp | -118 | 52 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 41.25% | 39.58% | 9.77 pp | -118 | 52 | -2.27 |

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
