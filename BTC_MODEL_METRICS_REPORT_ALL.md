# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T22:39:35.468446+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1267 | 979 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1142 | 777 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 21:00:00+00:00 | 846 | 539 | 306 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 21:00:00+00:00 | 848 | 593 | 253 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 539 | 262 | 277 | 48.61% | 46.25% | 48.33% | 1.39 pp | -15 | 51 | -0.29 |
| BTC Market Hours | transformer | Transformer | 539 | 258 | 281 | 47.87% | 49.17% | 48.33% | 2.13 pp | -23 | 51 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 767 | 371 | 396 | 48.37% | 47.08% | 48.54% | 1.63 pp | -25 | 45 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 593 | 282 | 311 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours | nn | NN | 539 | 255 | 284 | 47.31% | 50.42% | 48.96% | 2.69 pp | -29 | 51 | -0.57 |
| BTC Market Hours Daily | nn | NN | 593 | 276 | 317 | 46.54% | 46.25% | 48.12% | 3.46 pp | -41 | 51 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 945 | 452 | 493 | 47.83% | 50.42% | 47.29% | 2.17 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 593 | 275 | 318 | 46.37% | 50.83% | 47.29% | 3.63 pp | -43 | 51 | -0.84 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 767 | 359 | 408 | 46.81% | 41.67% | 47.08% | 3.19 pp | -49 | 45 | -1.09 |
| BTC Hourly | transformer | Transformer | 945 | 442 | 503 | 46.77% | 46.25% | 45.00% | 3.23 pp | -61 | 49 | -1.24 |
| BTC Daily | nn | NN | 767 | 355 | 412 | 46.28% | 44.58% | 46.04% | 3.72 pp | -57 | 45 | -1.27 |
| BTC Market Hours | rf | RandomForest | 539 | 233 | 306 | 43.23% | 45.42% | 43.33% | 6.77 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 539 | 232 | 307 | 43.04% | 42.08% | 43.96% | 6.96 pp | -75 | 51 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 539 | 224 | 315 | 41.56% | 43.33% | 41.88% | 8.44 pp | -91 | 51 | -1.78 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 593 | 248 | 345 | 41.82% | 45.42% | 41.67% | 8.18 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 945 | 420 | 525 | 44.44% | 42.92% | 42.92% | 5.56 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 945 | 419 | 526 | 44.34% | 44.17% | 43.75% | 5.66 pp | -107 | 49 | -2.18 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 593 | 240 | 353 | 40.47% | 39.17% | 40.00% | 9.53 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 593 | 236 | 357 | 39.80% | 41.25% | 38.96% | 10.20 pp | -121 | 51 | -2.37 |
| BTC Daily | lstm | LSTM | 767 | 325 | 442 | 42.37% | 36.25% | 40.62% | 7.63 pp | -117 | 45 | -2.60 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 767 | 321 | 446 | 41.85% | 38.33% | 42.29% | 8.15 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 945 | 404 | 541 | 42.75% | 36.67% | 42.29% | 7.25 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 945 | 396 | 549 | 41.90% | 40.42% | 40.62% | 8.10 pp | -153 | 49 | -3.12 |
| BTC Daily | xgb | XGBoost | 777 | 304 | 473 | 39.12% | 35.42% | 36.46% | 10.88 pp | -169 | 45 | -3.76 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 945 | 452 | 493 | 47.83% | 50.42% | 47.29% | 2.17 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 945 | 442 | 503 | 46.77% | 46.25% | 45.00% | 3.23 pp | -61 | 49 | -1.24 |
| BTC Hourly | nn | NN | 945 | 420 | 525 | 44.44% | 42.92% | 42.92% | 5.56 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 945 | 419 | 526 | 44.34% | 44.17% | 43.75% | 5.66 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 945 | 404 | 541 | 42.75% | 36.67% | 42.29% | 7.25 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 945 | 396 | 549 | 41.90% | 40.42% | 40.62% | 8.10 pp | -153 | 49 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 767 | 371 | 396 | 48.37% | 47.08% | 48.54% | 1.63 pp | -25 | 45 | -0.56 |
| BTC Daily | transformer | Transformer | 767 | 359 | 408 | 46.81% | 41.67% | 47.08% | 3.19 pp | -49 | 45 | -1.09 |
| BTC Daily | nn | NN | 767 | 355 | 412 | 46.28% | 44.58% | 46.04% | 3.72 pp | -57 | 45 | -1.27 |
| BTC Daily | lstm | LSTM | 767 | 325 | 442 | 42.37% | 36.25% | 40.62% | 7.63 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 767 | 321 | 446 | 41.85% | 38.33% | 42.29% | 8.15 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 777 | 304 | 473 | 39.12% | 35.42% | 36.46% | 10.88 pp | -169 | 45 | -3.76 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 539 | 262 | 277 | 48.61% | 46.25% | 48.33% | 1.39 pp | -15 | 51 | -0.29 |
| BTC Market Hours | transformer | Transformer | 539 | 258 | 281 | 47.87% | 49.17% | 48.33% | 2.13 pp | -23 | 51 | -0.45 |
| BTC Market Hours | nn | NN | 539 | 255 | 284 | 47.31% | 50.42% | 48.96% | 2.69 pp | -29 | 51 | -0.57 |
| BTC Market Hours | rf | RandomForest | 539 | 233 | 306 | 43.23% | 45.42% | 43.33% | 6.77 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 539 | 232 | 307 | 43.04% | 42.08% | 43.96% | 6.96 pp | -75 | 51 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 539 | 224 | 315 | 41.56% | 43.33% | 41.88% | 8.44 pp | -91 | 51 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 593 | 282 | 311 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours Daily | nn | NN | 593 | 276 | 317 | 46.54% | 46.25% | 48.12% | 3.46 pp | -41 | 51 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 593 | 275 | 318 | 46.37% | 50.83% | 47.29% | 3.63 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 593 | 248 | 345 | 41.82% | 45.42% | 41.67% | 8.18 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 593 | 240 | 353 | 40.47% | 39.17% | 40.00% | 9.53 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 593 | 236 | 357 | 39.80% | 41.25% | 38.96% | 10.20 pp | -121 | 51 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
