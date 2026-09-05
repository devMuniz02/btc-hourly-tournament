# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T01:18:07.486375+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1111 | 746 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 792 | 508 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 794 | 562 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 508 | 246 | 262 | 48.43% | 45.00% | 48.33% | 1.57 pp | -16 | 48 | -0.33 |
| BTC Market Hours | transformer | Transformer | 508 | 243 | 265 | 47.83% | 46.25% | 48.12% | 2.17 pp | -22 | 48 | -0.46 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 562 | 267 | 295 | 47.51% | 50.83% | 48.75% | 2.49 pp | -28 | 48 | -0.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 736 | 355 | 381 | 48.23% | 46.67% | 47.92% | 1.77 pp | -26 | 44 | -0.59 |
| BTC Market Hours | nn | NN | 508 | 239 | 269 | 47.05% | 49.17% | 47.92% | 2.95 pp | -30 | 48 | -0.62 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 914 | 438 | 476 | 47.92% | 50.42% | 47.71% | 2.08 pp | -38 | 48 | -0.79 |
| BTC Daily | transformer | Transformer | 736 | 350 | 386 | 47.55% | 45.83% | 49.58% | 2.45 pp | -36 | 44 | -0.82 |
| BTC Market Hours Daily | nn | NN | 562 | 261 | 301 | 46.44% | 45.83% | 47.92% | 3.56 pp | -40 | 48 | -0.83 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 562 | 259 | 303 | 46.09% | 48.75% | 46.67% | 3.91 pp | -44 | 48 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 914 | 432 | 482 | 47.26% | 47.92% | 46.46% | 2.74 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 736 | 341 | 395 | 46.33% | 44.58% | 47.08% | 3.67 pp | -54 | 44 | -1.23 |
| BTC Market Hours | lstm | LSTM | 508 | 221 | 287 | 43.50% | 42.92% | 43.54% | 6.50 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours | rf | RandomForest | 508 | 219 | 289 | 43.11% | 43.75% | 43.33% | 6.89 pp | -70 | 48 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 508 | 208 | 300 | 40.94% | 42.08% | 41.46% | 9.06 pp | -92 | 48 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 562 | 233 | 329 | 41.46% | 42.08% | 40.42% | 8.54 pp | -96 | 48 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | nn | NN | 914 | 406 | 508 | 44.42% | 43.33% | 42.08% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 914 | 406 | 508 | 44.42% | 43.75% | 43.96% | 5.58 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 562 | 228 | 334 | 40.57% | 39.17% | 40.83% | 9.43 pp | -106 | 48 | -2.21 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Daily | lstm | LSTM | 736 | 316 | 420 | 42.93% | 36.67% | 41.25% | 7.07 pp | -104 | 44 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 562 | 224 | 338 | 39.86% | 40.00% | 38.75% | 10.14 pp | -114 | 48 | -2.38 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 736 | 311 | 425 | 42.26% | 39.58% | 42.71% | 7.74 pp | -114 | 44 | -2.59 |
| BTC Hourly | lstm | LSTM | 914 | 391 | 523 | 42.78% | 39.58% | 41.46% | 7.22 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 914 | 382 | 532 | 41.79% | 40.00% | 40.21% | 8.21 pp | -150 | 48 | -3.12 |
| BTC Daily | xgb | XGBoost | 746 | 294 | 452 | 39.41% | 36.25% | 37.71% | 10.59 pp | -158 | 44 | -3.59 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 736 | 355 | 381 | 48.23% | 46.67% | 47.92% | 1.77 pp | -26 | 44 | -0.59 |
| BTC Daily | transformer | Transformer | 736 | 350 | 386 | 47.55% | 45.83% | 49.58% | 2.45 pp | -36 | 44 | -0.82 |
| BTC Daily | nn | NN | 736 | 341 | 395 | 46.33% | 44.58% | 47.08% | 3.67 pp | -54 | 44 | -1.23 |
| BTC Daily | lstm | LSTM | 736 | 316 | 420 | 42.93% | 36.67% | 41.25% | 7.07 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 736 | 311 | 425 | 42.26% | 39.58% | 42.71% | 7.74 pp | -114 | 44 | -2.59 |
| BTC Daily | xgb | XGBoost | 746 | 294 | 452 | 39.41% | 36.25% | 37.71% | 10.59 pp | -158 | 44 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 508 | 246 | 262 | 48.43% | 45.00% | 48.33% | 1.57 pp | -16 | 48 | -0.33 |
| BTC Market Hours | transformer | Transformer | 508 | 243 | 265 | 47.83% | 46.25% | 48.12% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Market Hours | nn | NN | 508 | 239 | 269 | 47.05% | 49.17% | 47.92% | 2.95 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 508 | 221 | 287 | 43.50% | 42.92% | 43.54% | 6.50 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 508 | 219 | 289 | 43.11% | 43.75% | 43.33% | 6.89 pp | -70 | 48 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 508 | 208 | 300 | 40.94% | 42.08% | 41.46% | 9.06 pp | -92 | 48 | -1.92 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 562 | 267 | 295 | 47.51% | 50.83% | 48.75% | 2.49 pp | -28 | 48 | -0.58 |
| BTC Market Hours Daily | nn | NN | 562 | 261 | 301 | 46.44% | 45.83% | 47.92% | 3.56 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 562 | 259 | 303 | 46.09% | 48.75% | 46.67% | 3.91 pp | -44 | 48 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 562 | 233 | 329 | 41.46% | 42.08% | 40.42% | 8.54 pp | -96 | 48 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 562 | 228 | 334 | 40.57% | 39.17% | 40.83% | 9.43 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 562 | 224 | 338 | 39.86% | 40.00% | 38.75% | 10.14 pp | -114 | 48 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
