# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T01:55:54.125800+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1236 | 948 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1112 | 747 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 793 | 509 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 795 | 563 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 156 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 4 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 509 | 246 | 263 | 48.33% | 45.00% | 48.33% | 1.67 pp | -17 | 49 | -0.35 |
| BTC Market Hours | transformer | Transformer | 509 | 244 | 265 | 47.94% | 46.25% | 48.33% | 2.06 pp | -21 | 49 | -0.43 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 737 | 356 | 381 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 44 | -0.57 |
| BTC Market Hours Daily | transformer | Transformer | 563 | 267 | 296 | 47.42% | 50.83% | 48.75% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours | nn | NN | 509 | 239 | 270 | 46.95% | 49.17% | 47.92% | 3.05 pp | -31 | 49 | -0.63 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 914 | 438 | 476 | 47.92% | 50.42% | 47.71% | 2.08 pp | -38 | 48 | -0.79 |
| BTC Daily | transformer | Transformer | 737 | 351 | 386 | 47.63% | 46.25% | 49.58% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Market Hours Daily | nn | NN | 563 | 261 | 302 | 46.36% | 45.83% | 47.71% | 3.64 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 563 | 260 | 303 | 46.18% | 48.75% | 46.67% | 3.82 pp | -43 | 48 | -0.90 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 914 | 432 | 482 | 47.26% | 47.92% | 46.46% | 2.74 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 737 | 342 | 395 | 46.40% | 44.58% | 47.08% | 3.60 pp | -53 | 44 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Market Hours | lstm | LSTM | 509 | 221 | 288 | 43.42% | 42.92% | 43.54% | 6.58 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 509 | 219 | 290 | 43.03% | 43.75% | 43.33% | 6.97 pp | -71 | 49 | -1.45 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 17 | 24 | 41.46% | 41.46% | 41.46% | 8.54 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 509 | 209 | 300 | 41.06% | 42.50% | 41.67% | 8.94 pp | -91 | 49 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 563 | 234 | 329 | 41.56% | 42.50% | 40.42% | 8.44 pp | -95 | 48 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | nn | NN | 914 | 406 | 508 | 44.42% | 43.33% | 42.08% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 914 | 406 | 508 | 44.42% | 43.75% | 43.96% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 563 | 229 | 334 | 40.67% | 39.58% | 41.04% | 9.33 pp | -105 | 48 | -2.19 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | nn | NN | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 563 | 225 | 338 | 39.96% | 40.42% | 38.96% | 10.04 pp | -113 | 48 | -2.35 |
| BTC Daily | lstm | LSTM | 737 | 316 | 421 | 42.88% | 36.67% | 41.04% | 7.12 pp | -105 | 44 | -2.39 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 737 | 312 | 425 | 42.33% | 40.00% | 42.71% | 7.67 pp | -113 | 44 | -2.57 |
| BTC Hourly | lstm | LSTM | 914 | 391 | 523 | 42.78% | 39.58% | 41.46% | 7.22 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 914 | 382 | 532 | 41.79% | 40.00% | 40.21% | 8.21 pp | -150 | 48 | -3.12 |
| BTC Daily | xgb | XGBoost | 747 | 295 | 452 | 39.49% | 36.25% | 37.71% | 10.51 pp | -157 | 44 | -3.57 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 914 | 438 | 476 | 47.92% | 50.42% | 47.71% | 2.08 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 914 | 432 | 482 | 47.26% | 47.92% | 46.46% | 2.74 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 914 | 406 | 508 | 44.42% | 43.33% | 42.08% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 914 | 406 | 508 | 44.42% | 43.75% | 43.96% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 914 | 391 | 523 | 42.78% | 39.58% | 41.46% | 7.22 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 914 | 382 | 532 | 41.79% | 40.00% | 40.21% | 8.21 pp | -150 | 48 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 737 | 356 | 381 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 44 | -0.57 |
| BTC Daily | transformer | Transformer | 737 | 351 | 386 | 47.63% | 46.25% | 49.58% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 737 | 342 | 395 | 46.40% | 44.58% | 47.08% | 3.60 pp | -53 | 44 | -1.20 |
| BTC Daily | lstm | LSTM | 737 | 316 | 421 | 42.88% | 36.67% | 41.04% | 7.12 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 737 | 312 | 425 | 42.33% | 40.00% | 42.71% | 7.67 pp | -113 | 44 | -2.57 |
| BTC Daily | xgb | XGBoost | 747 | 295 | 452 | 39.49% | 36.25% | 37.71% | 10.51 pp | -157 | 44 | -3.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 509 | 246 | 263 | 48.33% | 45.00% | 48.33% | 1.67 pp | -17 | 49 | -0.35 |
| BTC Market Hours | transformer | Transformer | 509 | 244 | 265 | 47.94% | 46.25% | 48.33% | 2.06 pp | -21 | 49 | -0.43 |
| BTC Market Hours | nn | NN | 509 | 239 | 270 | 46.95% | 49.17% | 47.92% | 3.05 pp | -31 | 49 | -0.63 |
| BTC Market Hours | lstm | LSTM | 509 | 221 | 288 | 43.42% | 42.92% | 43.54% | 6.58 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 509 | 219 | 290 | 43.03% | 43.75% | 43.33% | 6.97 pp | -71 | 49 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 509 | 209 | 300 | 41.06% | 42.50% | 41.67% | 8.94 pp | -91 | 49 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 563 | 267 | 296 | 47.42% | 50.83% | 48.75% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours Daily | nn | NN | 563 | 261 | 302 | 46.36% | 45.83% | 47.71% | 3.64 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 563 | 260 | 303 | 46.18% | 48.75% | 46.67% | 3.82 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 563 | 234 | 329 | 41.56% | 42.50% | 40.42% | 8.44 pp | -95 | 48 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 563 | 229 | 334 | 40.67% | 39.58% | 41.04% | 9.33 pp | -105 | 48 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 563 | 225 | 338 | 39.96% | 40.42% | 38.96% | 10.04 pp | -113 | 48 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 4 | 0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 17 | 24 | 41.46% | 41.46% | 41.46% | 8.54 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | nn | NN | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
