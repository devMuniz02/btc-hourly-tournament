# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T07:12:56.963537+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1289 | 1001 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1165 | 800 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 885 | 562 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 886 | 615 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 203 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 203 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 66 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 66 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.08% | 47.92% | 1.25 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 562 | 267 | 295 | 47.51% | 51.25% | 49.17% | 2.49 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 382 | 408 | 48.35% | 47.50% | 48.33% | 1.65 pp | -26 | 46 | -0.57 |
| BTC Market Hours | transformer | Transformer | 562 | 264 | 298 | 46.98% | 46.25% | 47.29% | 3.02 pp | -34 | 53 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 615 | 287 | 328 | 46.67% | 49.17% | 47.29% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | nn | NN | 615 | 287 | 328 | 46.67% | 47.50% | 48.33% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 615 | 287 | 328 | 46.67% | 48.75% | 47.50% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 967 | 460 | 507 | 47.57% | 49.17% | 46.46% | 2.43 pp | -47 | 50 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 967 | 450 | 517 | 46.54% | 44.17% | 43.75% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 562 | 245 | 317 | 43.59% | 42.50% | 44.17% | 6.41 pp | -72 | 53 | -1.36 |
| BTC Market Hours | rf | RandomForest | 562 | 243 | 319 | 43.24% | 45.83% | 43.54% | 6.76 pp | -76 | 53 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 562 | 239 | 323 | 42.53% | 45.83% | 42.92% | 7.47 pp | -84 | 53 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 615 | 258 | 357 | 41.95% | 44.58% | 41.25% | 8.05 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 615 | 251 | 364 | 40.81% | 43.33% | 40.62% | 9.19 pp | -113 | 52 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 615 | 250 | 365 | 40.65% | 40.83% | 40.21% | 9.35 pp | -115 | 52 | -2.21 |
| BTC Hourly | rf | RandomForest | 967 | 428 | 539 | 44.26% | 42.08% | 42.50% | 5.74 pp | -111 | 50 | -2.22 |
| BTC Hourly | nn | NN | 967 | 427 | 540 | 44.16% | 41.25% | 42.50% | 5.84 pp | -113 | 50 | -2.26 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 331 | 459 | 41.90% | 38.75% | 41.67% | 8.10 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 967 | 412 | 555 | 42.61% | 37.50% | 41.46% | 7.39 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 967 | 399 | 568 | 41.26% | 36.67% | 38.75% | 8.74 pp | -169 | 50 | -3.38 |
| BTC Daily | xgb | XGBoost | 800 | 313 | 487 | 39.12% | 36.25% | 36.25% | 10.88 pp | -174 | 46 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 967 | 460 | 507 | 47.57% | 49.17% | 46.46% | 2.43 pp | -47 | 50 | -0.94 |
| BTC Hourly | transformer | Transformer | 967 | 450 | 517 | 46.54% | 44.17% | 43.75% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Hourly | rf | RandomForest | 967 | 428 | 539 | 44.26% | 42.08% | 42.50% | 5.74 pp | -111 | 50 | -2.22 |
| BTC Hourly | nn | NN | 967 | 427 | 540 | 44.16% | 41.25% | 42.50% | 5.84 pp | -113 | 50 | -2.26 |
| BTC Hourly | lstm | LSTM | 967 | 412 | 555 | 42.61% | 37.50% | 41.46% | 7.39 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 967 | 399 | 568 | 41.26% | 36.67% | 38.75% | 8.74 pp | -169 | 50 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 382 | 408 | 48.35% | 47.50% | 48.33% | 1.65 pp | -26 | 46 | -0.57 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 331 | 459 | 41.90% | 38.75% | 41.67% | 8.10 pp | -128 | 46 | -2.78 |
| BTC Daily | xgb | XGBoost | 800 | 313 | 487 | 39.12% | 36.25% | 36.25% | 10.88 pp | -174 | 46 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.08% | 47.92% | 1.25 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 562 | 267 | 295 | 47.51% | 51.25% | 49.17% | 2.49 pp | -28 | 53 | -0.53 |
| BTC Market Hours | transformer | Transformer | 562 | 264 | 298 | 46.98% | 46.25% | 47.29% | 3.02 pp | -34 | 53 | -0.64 |
| BTC Market Hours | lstm | LSTM | 562 | 245 | 317 | 43.59% | 42.50% | 44.17% | 6.41 pp | -72 | 53 | -1.36 |
| BTC Market Hours | rf | RandomForest | 562 | 243 | 319 | 43.24% | 45.83% | 43.54% | 6.76 pp | -76 | 53 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 562 | 239 | 323 | 42.53% | 45.83% | 42.92% | 7.47 pp | -84 | 53 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 615 | 287 | 328 | 46.67% | 49.17% | 47.29% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | nn | NN | 615 | 287 | 328 | 46.67% | 47.50% | 48.33% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 615 | 287 | 328 | 46.67% | 48.75% | 47.50% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 615 | 258 | 357 | 41.95% | 44.58% | 41.25% | 8.05 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 615 | 251 | 364 | 40.81% | 43.33% | 40.62% | 9.19 pp | -113 | 52 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 615 | 250 | 365 | 40.65% | 40.83% | 40.21% | 9.35 pp | -115 | 52 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
