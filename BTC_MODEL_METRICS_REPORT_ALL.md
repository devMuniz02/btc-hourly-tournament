# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T03:16:20.138564+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1286 | 998 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1162 | 797 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 882 | 559 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 884 | 613 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 203 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 559 | 273 | 286 | 48.84% | 47.50% | 47.92% | 1.16 pp | -13 | 52 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| BTC Market Hours | nn | NN | 559 | 266 | 293 | 47.58% | 51.67% | 49.38% | 2.42 pp | -27 | 52 | -0.52 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 787 | 379 | 408 | 48.16% | 46.25% | 47.92% | 1.84 pp | -29 | 46 | -0.63 |
| BTC Market Hours | transformer | Transformer | 559 | 263 | 296 | 47.05% | 46.25% | 47.29% | 2.95 pp | -33 | 52 | -0.63 |
| BTC Market Hours Daily | nn | NN | 613 | 287 | 326 | 46.82% | 47.92% | 48.33% | 3.18 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 613 | 287 | 326 | 46.82% | 49.17% | 47.71% | 3.18 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 613 | 286 | 327 | 46.66% | 49.17% | 47.50% | 3.34 pp | -41 | 52 | -0.79 |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 5 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 964 | 459 | 505 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| BTC Daily | transformer | Transformer | 787 | 366 | 421 | 46.51% | 40.42% | 46.46% | 3.49 pp | -55 | 46 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 787 | 364 | 423 | 46.25% | 44.58% | 45.00% | 3.75 pp | -59 | 46 | -1.28 |
| BTC Hourly | transformer | Transformer | 964 | 449 | 515 | 46.58% | 45.00% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 559 | 243 | 316 | 43.47% | 42.50% | 43.96% | 6.53 pp | -73 | 52 | -1.40 |
| BTC Market Hours | rf | RandomForest | 559 | 242 | 317 | 43.29% | 45.42% | 43.33% | 6.71 pp | -75 | 52 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 559 | 238 | 321 | 42.58% | 45.83% | 43.12% | 7.42 pp | -83 | 52 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 613 | 257 | 356 | 41.92% | 44.17% | 41.04% | 8.08 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 613 | 250 | 363 | 40.78% | 42.92% | 40.42% | 9.22 pp | -113 | 52 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 613 | 249 | 364 | 40.62% | 40.83% | 40.42% | 9.38 pp | -115 | 52 | -2.21 |
| BTC Hourly | nn | NN | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 787 | 331 | 456 | 42.06% | 33.75% | 39.79% | 7.94 pp | -125 | 46 | -2.72 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 5 | -2.80 |
| BTC Daily | rf | RandomForest | 787 | 329 | 458 | 41.80% | 38.33% | 41.67% | 8.20 pp | -129 | 46 | -2.80 |
| BTC Hourly | lstm | LSTM | 964 | 410 | 554 | 42.53% | 37.08% | 41.46% | 7.47 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 964 | 397 | 567 | 41.18% | 36.67% | 38.54% | 8.82 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 797 | 311 | 486 | 39.02% | 35.83% | 36.25% | 10.98 pp | -175 | 46 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 964 | 459 | 505 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 964 | 449 | 515 | 46.58% | 45.00% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| BTC Hourly | nn | NN | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | lstm | LSTM | 964 | 410 | 554 | 42.53% | 37.08% | 41.46% | 7.47 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 964 | 397 | 567 | 41.18% | 36.67% | 38.54% | 8.82 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 787 | 379 | 408 | 48.16% | 46.25% | 47.92% | 1.84 pp | -29 | 46 | -0.63 |
| BTC Daily | transformer | Transformer | 787 | 366 | 421 | 46.51% | 40.42% | 46.46% | 3.49 pp | -55 | 46 | -1.20 |
| BTC Daily | nn | NN | 787 | 364 | 423 | 46.25% | 44.58% | 45.00% | 3.75 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 787 | 331 | 456 | 42.06% | 33.75% | 39.79% | 7.94 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 787 | 329 | 458 | 41.80% | 38.33% | 41.67% | 8.20 pp | -129 | 46 | -2.80 |
| BTC Daily | xgb | XGBoost | 797 | 311 | 486 | 39.02% | 35.83% | 36.25% | 10.98 pp | -175 | 46 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 559 | 273 | 286 | 48.84% | 47.50% | 47.92% | 1.16 pp | -13 | 52 | -0.25 |
| BTC Market Hours | nn | NN | 559 | 266 | 293 | 47.58% | 51.67% | 49.38% | 2.42 pp | -27 | 52 | -0.52 |
| BTC Market Hours | transformer | Transformer | 559 | 263 | 296 | 47.05% | 46.25% | 47.29% | 2.95 pp | -33 | 52 | -0.63 |
| BTC Market Hours | lstm | LSTM | 559 | 243 | 316 | 43.47% | 42.50% | 43.96% | 6.53 pp | -73 | 52 | -1.40 |
| BTC Market Hours | rf | RandomForest | 559 | 242 | 317 | 43.29% | 45.42% | 43.33% | 6.71 pp | -75 | 52 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 559 | 238 | 321 | 42.58% | 45.83% | 43.12% | 7.42 pp | -83 | 52 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 613 | 287 | 326 | 46.82% | 47.92% | 48.33% | 3.18 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 613 | 287 | 326 | 46.82% | 49.17% | 47.71% | 3.18 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 613 | 286 | 327 | 46.66% | 49.17% | 47.50% | 3.34 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 613 | 257 | 356 | 41.92% | 44.17% | 41.04% | 8.08 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 613 | 250 | 363 | 40.78% | 42.92% | 40.42% | 9.22 pp | -113 | 52 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 613 | 249 | 364 | 40.62% | 40.83% | 40.42% | 9.38 pp | -115 | 52 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
