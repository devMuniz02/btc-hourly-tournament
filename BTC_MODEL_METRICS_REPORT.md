# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T09:31:07.771226+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1338 | 1050 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1213 | 848 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 972 | 610 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 974 | 664 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 610 | 294 | 316 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 610 | 291 | 319 | 47.70% | 50.42% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours Daily | nn | NN | 664 | 313 | 351 | 47.14% | 50.00% | 48.12% | 2.86 pp | -38 | 56 | -0.68 |
| BTC Market Hours | transformer | Transformer | 610 | 285 | 325 | 46.72% | 46.25% | 45.62% | 3.28 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 664 | 311 | 353 | 46.84% | 49.17% | 47.08% | 3.16 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 664 | 309 | 355 | 46.54% | 48.33% | 47.92% | 3.46 pp | -46 | 56 | -0.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 838 | 399 | 439 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1016 | 481 | 535 | 47.34% | 47.92% | 46.04% | 2.66 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 838 | 390 | 448 | 46.54% | 45.00% | 45.21% | 3.46 pp | -58 | 48 | -1.21 |
| BTC Hourly | transformer | Transformer | 1016 | 475 | 541 | 46.75% | 46.25% | 44.79% | 3.25 pp | -66 | 52 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| BTC Daily | transformer | Transformer | 838 | 385 | 453 | 45.94% | 37.50% | 43.96% | 4.06 pp | -68 | 48 | -1.42 |
| BTC Market Hours | lstm | LSTM | 610 | 261 | 349 | 42.79% | 42.92% | 42.92% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 610 | 261 | 349 | 42.79% | 43.33% | 42.08% | 7.21 pp | -88 | 56 | -1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 610 | 259 | 351 | 42.46% | 45.83% | 43.33% | 7.54 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 664 | 276 | 388 | 41.57% | 43.75% | 41.67% | 8.43 pp | -112 | 56 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 664 | 272 | 392 | 40.96% | 44.17% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 664 | 272 | 392 | 40.96% | 43.33% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1016 | 447 | 569 | 44.00% | 41.25% | 40.62% | 6.00 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1016 | 445 | 571 | 43.80% | 40.42% | 42.08% | 6.20 pp | -126 | 52 | -2.42 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| BTC Daily | lstm | LSTM | 838 | 353 | 485 | 42.12% | 35.42% | 39.79% | 7.88 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| BTC Daily | rf | RandomForest | 838 | 347 | 491 | 41.41% | 36.67% | 40.62% | 8.59 pp | -144 | 48 | -3.00 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| BTC Hourly | lstm | LSTM | 1016 | 428 | 588 | 42.13% | 35.00% | 39.38% | 7.87 pp | -160 | 52 | -3.08 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| BTC Hourly | xgb | XGBoost | 1016 | 415 | 601 | 40.85% | 34.17% | 37.50% | 9.15 pp | -186 | 52 | -3.58 |
| BTC Daily | xgb | XGBoost | 848 | 334 | 514 | 39.39% | 37.50% | 36.67% | 10.61 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1016 | 481 | 535 | 47.34% | 47.92% | 46.04% | 2.66 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1016 | 475 | 541 | 46.75% | 46.25% | 44.79% | 3.25 pp | -66 | 52 | -1.27 |
| BTC Hourly | nn | NN | 1016 | 447 | 569 | 44.00% | 41.25% | 40.62% | 6.00 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1016 | 445 | 571 | 43.80% | 40.42% | 42.08% | 6.20 pp | -126 | 52 | -2.42 |
| BTC Hourly | lstm | LSTM | 1016 | 428 | 588 | 42.13% | 35.00% | 39.38% | 7.87 pp | -160 | 52 | -3.08 |
| BTC Hourly | xgb | XGBoost | 1016 | 415 | 601 | 40.85% | 34.17% | 37.50% | 9.15 pp | -186 | 52 | -3.58 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 838 | 399 | 439 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| BTC Daily | nn | NN | 838 | 390 | 448 | 46.54% | 45.00% | 45.21% | 3.46 pp | -58 | 48 | -1.21 |
| BTC Daily | transformer | Transformer | 838 | 385 | 453 | 45.94% | 37.50% | 43.96% | 4.06 pp | -68 | 48 | -1.42 |
| BTC Daily | lstm | LSTM | 838 | 353 | 485 | 42.12% | 35.42% | 39.79% | 7.88 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 838 | 347 | 491 | 41.41% | 36.67% | 40.62% | 8.59 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 848 | 334 | 514 | 39.39% | 37.50% | 36.67% | 10.61 pp | -180 | 48 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 610 | 294 | 316 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 610 | 291 | 319 | 47.70% | 50.42% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 610 | 285 | 325 | 46.72% | 46.25% | 45.62% | 3.28 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 610 | 261 | 349 | 42.79% | 42.92% | 42.92% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 610 | 261 | 349 | 42.79% | 43.33% | 42.08% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 610 | 259 | 351 | 42.46% | 45.83% | 43.33% | 7.54 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 664 | 313 | 351 | 47.14% | 50.00% | 48.12% | 2.86 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 664 | 311 | 353 | 46.84% | 49.17% | 47.08% | 3.16 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 664 | 309 | 355 | 46.54% | 48.33% | 47.92% | 3.46 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 664 | 276 | 388 | 41.57% | 43.75% | 41.67% | 8.43 pp | -112 | 56 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 664 | 272 | 392 | 40.96% | 44.17% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 664 | 272 | 392 | 40.96% | 43.33% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
