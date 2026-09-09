# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T15:54:50.957455+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1311 | 1023 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1186 | 821 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 922 | 583 | 338 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 924 | 637 | 285 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 811 | 410 | 401 | 50.55% | 47.50% | 49.17% | 0.55 pp | 9 | 47 | 0.19 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 583 | 296 | 287 | 50.77% | 48.75% | 49.79% | 0.77 pp | 9 | 54 | 0.17 |
| BTC Market Hours | nn | NN | 583 | 289 | 294 | 49.57% | 52.08% | 50.00% | 0.43 pp | -5 | 54 | -0.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 989 | 489 | 500 | 49.44% | 50.83% | 47.71% | 0.56 pp | -11 | 51 | -0.22 |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| BTC Market Hours Daily | nn | NN | 637 | 300 | 337 | 47.10% | 48.33% | 48.33% | 2.90 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 637 | 299 | 338 | 46.94% | 48.75% | 47.50% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 637 | 299 | 338 | 46.94% | 49.58% | 47.29% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours | transformer | Transformer | 583 | 269 | 314 | 46.14% | 45.00% | 45.83% | 3.86 pp | -45 | 54 | -0.83 |
| BTC Daily | nn | NN | 811 | 381 | 430 | 46.98% | 44.58% | 45.42% | 3.02 pp | -49 | 47 | -1.04 |
| BTC Market Hours | rf | RandomForest | 583 | 259 | 324 | 44.43% | 45.83% | 44.17% | 5.57 pp | -65 | 54 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Hourly | transformer | Transformer | 989 | 457 | 532 | 46.21% | 43.33% | 42.29% | 3.79 pp | -75 | 51 | -1.47 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 811 | 368 | 443 | 45.38% | 39.17% | 44.79% | 4.62 pp | -75 | 47 | -1.60 |
| BTC Hourly | nn | NN | 989 | 453 | 536 | 45.80% | 42.92% | 42.71% | 4.20 pp | -83 | 51 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 583 | 246 | 337 | 42.20% | 42.92% | 41.88% | 7.80 pp | -91 | 54 | -1.69 |
| BTC Hourly | rf | RandomForest | 989 | 449 | 540 | 45.40% | 40.42% | 42.50% | 4.60 pp | -91 | 51 | -1.78 |
| BTC Market Hours | lstm | LSTM | 583 | 243 | 340 | 41.68% | 37.08% | 40.62% | 8.32 pp | -97 | 54 | -1.80 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 637 | 266 | 371 | 41.76% | 42.92% | 41.04% | 8.24 pp | -105 | 54 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 637 | 263 | 374 | 41.29% | 44.17% | 40.62% | 8.71 pp | -111 | 54 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 637 | 261 | 376 | 40.97% | 41.67% | 40.62% | 9.03 pp | -115 | 54 | -2.13 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| BTC Daily | lstm | LSTM | 811 | 353 | 458 | 43.53% | 35.83% | 41.25% | 6.47 pp | -105 | 47 | -2.23 |
| BTC Hourly | lstm | LSTM | 989 | 435 | 554 | 43.98% | 39.17% | 41.67% | 6.02 pp | -119 | 51 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Daily | rf | RandomForest | 811 | 346 | 465 | 42.66% | 36.67% | 41.25% | 7.34 pp | -119 | 47 | -2.53 |
| BTC Hourly | xgb | XGBoost | 989 | 419 | 570 | 42.37% | 35.83% | 39.79% | 7.63 pp | -151 | 51 | -2.96 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 821 | 322 | 499 | 39.22% | 35.42% | 36.25% | 10.78 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 989 | 489 | 500 | 49.44% | 50.83% | 47.71% | 0.56 pp | -11 | 51 | -0.22 |
| BTC Hourly | transformer | Transformer | 989 | 457 | 532 | 46.21% | 43.33% | 42.29% | 3.79 pp | -75 | 51 | -1.47 |
| BTC Hourly | nn | NN | 989 | 453 | 536 | 45.80% | 42.92% | 42.71% | 4.20 pp | -83 | 51 | -1.63 |
| BTC Hourly | rf | RandomForest | 989 | 449 | 540 | 45.40% | 40.42% | 42.50% | 4.60 pp | -91 | 51 | -1.78 |
| BTC Hourly | lstm | LSTM | 989 | 435 | 554 | 43.98% | 39.17% | 41.67% | 6.02 pp | -119 | 51 | -2.33 |
| BTC Hourly | xgb | XGBoost | 989 | 419 | 570 | 42.37% | 35.83% | 39.79% | 7.63 pp | -151 | 51 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 811 | 410 | 401 | 50.55% | 47.50% | 49.17% | 0.55 pp | 9 | 47 | 0.19 |
| BTC Daily | nn | NN | 811 | 381 | 430 | 46.98% | 44.58% | 45.42% | 3.02 pp | -49 | 47 | -1.04 |
| BTC Daily | transformer | Transformer | 811 | 368 | 443 | 45.38% | 39.17% | 44.79% | 4.62 pp | -75 | 47 | -1.60 |
| BTC Daily | lstm | LSTM | 811 | 353 | 458 | 43.53% | 35.83% | 41.25% | 6.47 pp | -105 | 47 | -2.23 |
| BTC Daily | rf | RandomForest | 811 | 346 | 465 | 42.66% | 36.67% | 41.25% | 7.34 pp | -119 | 47 | -2.53 |
| BTC Daily | xgb | XGBoost | 821 | 322 | 499 | 39.22% | 35.42% | 36.25% | 10.78 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 583 | 296 | 287 | 50.77% | 48.75% | 49.79% | 0.77 pp | 9 | 54 | 0.17 |
| BTC Market Hours | nn | NN | 583 | 289 | 294 | 49.57% | 52.08% | 50.00% | 0.43 pp | -5 | 54 | -0.09 |
| BTC Market Hours | transformer | Transformer | 583 | 269 | 314 | 46.14% | 45.00% | 45.83% | 3.86 pp | -45 | 54 | -0.83 |
| BTC Market Hours | rf | RandomForest | 583 | 259 | 324 | 44.43% | 45.83% | 44.17% | 5.57 pp | -65 | 54 | -1.20 |
| BTC Market Hours | xgb | XGBoost | 583 | 246 | 337 | 42.20% | 42.92% | 41.88% | 7.80 pp | -91 | 54 | -1.69 |
| BTC Market Hours | lstm | LSTM | 583 | 243 | 340 | 41.68% | 37.08% | 40.62% | 8.32 pp | -97 | 54 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 637 | 300 | 337 | 47.10% | 48.33% | 48.33% | 2.90 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 637 | 299 | 338 | 46.94% | 48.75% | 47.50% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 637 | 299 | 338 | 46.94% | 49.58% | 47.29% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | rf | RandomForest | 637 | 266 | 371 | 41.76% | 42.92% | 41.04% | 8.24 pp | -105 | 54 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 637 | 263 | 374 | 41.29% | 44.17% | 40.62% | 8.71 pp | -111 | 54 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 637 | 261 | 376 | 40.97% | 41.67% | 40.62% | 9.03 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
