# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T04:36:14.630149+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1271 | 983 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1147 | 782 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 854 | 544 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 855 | 597 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 544 | 275 | 269 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| BTC Market Hours | nn | NN | 544 | 269 | 275 | 49.45% | 51.67% | 51.04% | 0.55 pp | -6 | 51 | -0.12 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 597 | 285 | 312 | 47.74% | 47.08% | 48.12% | 2.26 pp | -27 | 51 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 772 | 373 | 399 | 48.32% | 46.67% | 48.12% | 1.68 pp | -26 | 45 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 597 | 281 | 316 | 47.07% | 47.92% | 46.88% | 2.93 pp | -35 | 51 | -0.69 |
| BTC Market Hours | transformer | Transformer | 544 | 253 | 291 | 46.51% | 46.25% | 47.29% | 3.49 pp | -38 | 51 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 949 | 453 | 496 | 47.73% | 49.58% | 47.08% | 2.27 pp | -43 | 50 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 597 | 276 | 321 | 46.23% | 47.92% | 47.50% | 3.77 pp | -45 | 51 | -0.88 |
| BTC Daily | transformer | Transformer | 772 | 360 | 412 | 46.63% | 41.25% | 46.88% | 3.37 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 544 | 241 | 303 | 44.30% | 46.67% | 44.17% | 5.70 pp | -62 | 51 | -1.22 |
| BTC Daily | nn | NN | 772 | 357 | 415 | 46.24% | 44.17% | 45.62% | 3.76 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 949 | 442 | 507 | 46.58% | 45.42% | 44.38% | 3.42 pp | -65 | 50 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 597 | 263 | 334 | 44.05% | 46.25% | 43.54% | 5.95 pp | -71 | 51 | -1.39 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 544 | 226 | 318 | 41.54% | 36.67% | 41.67% | 8.46 pp | -92 | 51 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 544 | 225 | 319 | 41.36% | 42.50% | 41.67% | 8.64 pp | -94 | 51 | -1.84 |
| BTC Market Hours Daily | xgb | XGBoost | 597 | 250 | 347 | 41.88% | 42.50% | 41.46% | 8.12 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 949 | 422 | 527 | 44.47% | 44.58% | 43.75% | 5.53 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 949 | 420 | 529 | 44.26% | 42.08% | 42.71% | 5.74 pp | -109 | 50 | -2.18 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 597 | 237 | 360 | 39.70% | 35.83% | 39.38% | 10.30 pp | -123 | 51 | -2.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 772 | 325 | 447 | 42.10% | 35.42% | 40.21% | 7.90 pp | -122 | 45 | -2.71 |
| BTC Hourly | lstm | LSTM | 949 | 406 | 543 | 42.78% | 36.67% | 42.29% | 7.22 pp | -137 | 50 | -2.74 |
| BTC Daily | rf | RandomForest | 772 | 323 | 449 | 41.84% | 38.33% | 42.08% | 8.16 pp | -126 | 45 | -2.80 |
| BTC Hourly | xgb | XGBoost | 949 | 396 | 553 | 41.73% | 39.58% | 40.42% | 8.27 pp | -157 | 50 | -3.14 |
| BTC Daily | xgb | XGBoost | 782 | 305 | 477 | 39.00% | 34.58% | 36.46% | 11.00 pp | -172 | 45 | -3.82 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 949 | 453 | 496 | 47.73% | 49.58% | 47.08% | 2.27 pp | -43 | 50 | -0.86 |
| BTC Hourly | transformer | Transformer | 949 | 442 | 507 | 46.58% | 45.42% | 44.38% | 3.42 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 949 | 422 | 527 | 44.47% | 44.58% | 43.75% | 5.53 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 949 | 420 | 529 | 44.26% | 42.08% | 42.71% | 5.74 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 949 | 406 | 543 | 42.78% | 36.67% | 42.29% | 7.22 pp | -137 | 50 | -2.74 |
| BTC Hourly | xgb | XGBoost | 949 | 396 | 553 | 41.73% | 39.58% | 40.42% | 8.27 pp | -157 | 50 | -3.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 772 | 373 | 399 | 48.32% | 46.67% | 48.12% | 1.68 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 772 | 360 | 412 | 46.63% | 41.25% | 46.88% | 3.37 pp | -52 | 45 | -1.16 |
| BTC Daily | nn | NN | 772 | 357 | 415 | 46.24% | 44.17% | 45.62% | 3.76 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 772 | 325 | 447 | 42.10% | 35.42% | 40.21% | 7.90 pp | -122 | 45 | -2.71 |
| BTC Daily | rf | RandomForest | 772 | 323 | 449 | 41.84% | 38.33% | 42.08% | 8.16 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 782 | 305 | 477 | 39.00% | 34.58% | 36.46% | 11.00 pp | -172 | 45 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 544 | 275 | 269 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| BTC Market Hours | nn | NN | 544 | 269 | 275 | 49.45% | 51.67% | 51.04% | 0.55 pp | -6 | 51 | -0.12 |
| BTC Market Hours | transformer | Transformer | 544 | 253 | 291 | 46.51% | 46.25% | 47.29% | 3.49 pp | -38 | 51 | -0.75 |
| BTC Market Hours | rf | RandomForest | 544 | 241 | 303 | 44.30% | 46.67% | 44.17% | 5.70 pp | -62 | 51 | -1.22 |
| BTC Market Hours | lstm | LSTM | 544 | 226 | 318 | 41.54% | 36.67% | 41.67% | 8.46 pp | -92 | 51 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 544 | 225 | 319 | 41.36% | 42.50% | 41.67% | 8.64 pp | -94 | 51 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 597 | 285 | 312 | 47.74% | 47.08% | 48.12% | 2.26 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 597 | 281 | 316 | 47.07% | 47.92% | 46.88% | 2.93 pp | -35 | 51 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 597 | 276 | 321 | 46.23% | 47.92% | 47.50% | 3.77 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 597 | 263 | 334 | 44.05% | 46.25% | 43.54% | 5.95 pp | -71 | 51 | -1.39 |
| BTC Market Hours Daily | xgb | XGBoost | 597 | 250 | 347 | 41.88% | 42.50% | 41.46% | 8.12 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 597 | 237 | 360 | 39.70% | 35.83% | 39.38% | 10.30 pp | -123 | 51 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
