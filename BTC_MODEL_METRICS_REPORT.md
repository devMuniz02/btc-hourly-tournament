# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T13:05:35.573667+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1324 | 1036 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1200 | 835 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 947 | 597 | 349 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 949 | 651 | 296 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 237 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 237 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 237 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 238 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 597 | 287 | 310 | 48.07% | 45.83% | 47.08% | 1.93 pp | -23 | 55 | -0.42 |
| BTC Market Hours | nn | NN | 597 | 287 | 310 | 48.07% | 51.67% | 49.79% | 1.93 pp | -23 | 55 | -0.42 |
| BTC Market Hours | transformer | Transformer | 597 | 280 | 317 | 46.90% | 46.25% | 46.25% | 3.10 pp | -37 | 55 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 825 | 395 | 430 | 47.88% | 45.00% | 46.67% | 2.12 pp | -35 | 47 | -0.74 |
| BTC Market Hours Daily | nn | NN | 651 | 305 | 346 | 46.85% | 48.33% | 47.71% | 3.15 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 651 | 304 | 347 | 46.70% | 48.33% | 47.29% | 3.30 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 651 | 304 | 347 | 46.70% | 47.92% | 48.12% | 3.30 pp | -43 | 55 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1002 | 475 | 527 | 47.41% | 48.75% | 46.25% | 2.59 pp | -52 | 52 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 15 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 825 | 385 | 440 | 46.67% | 45.42% | 45.42% | 3.33 pp | -55 | 47 | -1.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Daily | transformer | Transformer | 825 | 382 | 443 | 46.30% | 37.92% | 45.00% | 3.70 pp | -61 | 47 | -1.30 |
| BTC Hourly | transformer | Transformer | 1002 | 466 | 536 | 46.51% | 45.83% | 44.79% | 3.49 pp | -70 | 52 | -1.35 |
| Consolidated Hourly | lstm | LSTM | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 597 | 256 | 341 | 42.88% | 42.08% | 42.71% | 7.12 pp | -85 | 55 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 597 | 255 | 342 | 42.71% | 45.00% | 43.54% | 7.29 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 597 | 254 | 343 | 42.55% | 42.08% | 42.08% | 7.45 pp | -89 | 55 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 651 | 269 | 382 | 41.32% | 41.67% | 41.04% | 8.68 pp | -113 | 55 | -2.05 |
| Consolidated Hourly | transformer | Transformer | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 651 | 268 | 383 | 41.17% | 42.92% | 40.62% | 8.83 pp | -115 | 55 | -2.09 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 651 | 266 | 385 | 40.86% | 42.50% | 40.21% | 9.14 pp | -119 | 55 | -2.16 |
| BTC Hourly | nn | NN | 1002 | 443 | 559 | 44.21% | 42.50% | 41.88% | 5.79 pp | -116 | 52 | -2.23 |
| BTC Hourly | rf | RandomForest | 1002 | 442 | 560 | 44.11% | 42.08% | 43.54% | 5.89 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| Consolidated Hourly | nn | NN | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 15 | -2.73 |
| BTC Daily | lstm | LSTM | 825 | 348 | 477 | 42.18% | 35.00% | 40.42% | 7.82 pp | -129 | 47 | -2.74 |
| BTC Hourly | lstm | LSTM | 1002 | 425 | 577 | 42.42% | 37.08% | 40.21% | 7.58 pp | -152 | 52 | -2.92 |
| BTC Daily | rf | RandomForest | 825 | 343 | 482 | 41.58% | 37.50% | 40.83% | 8.42 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1002 | 413 | 589 | 41.22% | 35.83% | 38.96% | 8.78 pp | -176 | 52 | -3.38 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 835 | 329 | 506 | 39.40% | 37.08% | 36.46% | 10.60 pp | -177 | 47 | -3.77 |
| Consolidated Market Hours Daily | nn | NN | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 7 | -3.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1002 | 475 | 527 | 47.41% | 48.75% | 46.25% | 2.59 pp | -52 | 52 | -1.00 |
| BTC Hourly | transformer | Transformer | 1002 | 466 | 536 | 46.51% | 45.83% | 44.79% | 3.49 pp | -70 | 52 | -1.35 |
| BTC Hourly | nn | NN | 1002 | 443 | 559 | 44.21% | 42.50% | 41.88% | 5.79 pp | -116 | 52 | -2.23 |
| BTC Hourly | rf | RandomForest | 1002 | 442 | 560 | 44.11% | 42.08% | 43.54% | 5.89 pp | -118 | 52 | -2.27 |
| BTC Hourly | lstm | LSTM | 1002 | 425 | 577 | 42.42% | 37.08% | 40.21% | 7.58 pp | -152 | 52 | -2.92 |
| BTC Hourly | xgb | XGBoost | 1002 | 413 | 589 | 41.22% | 35.83% | 38.96% | 8.78 pp | -176 | 52 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 825 | 395 | 430 | 47.88% | 45.00% | 46.67% | 2.12 pp | -35 | 47 | -0.74 |
| BTC Daily | nn | NN | 825 | 385 | 440 | 46.67% | 45.42% | 45.42% | 3.33 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 825 | 382 | 443 | 46.30% | 37.92% | 45.00% | 3.70 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 825 | 348 | 477 | 42.18% | 35.00% | 40.42% | 7.82 pp | -129 | 47 | -2.74 |
| BTC Daily | rf | RandomForest | 825 | 343 | 482 | 41.58% | 37.50% | 40.83% | 8.42 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 835 | 329 | 506 | 39.40% | 37.08% | 36.46% | 10.60 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 597 | 287 | 310 | 48.07% | 45.83% | 47.08% | 1.93 pp | -23 | 55 | -0.42 |
| BTC Market Hours | nn | NN | 597 | 287 | 310 | 48.07% | 51.67% | 49.79% | 1.93 pp | -23 | 55 | -0.42 |
| BTC Market Hours | transformer | Transformer | 597 | 280 | 317 | 46.90% | 46.25% | 46.25% | 3.10 pp | -37 | 55 | -0.67 |
| BTC Market Hours | lstm | LSTM | 597 | 256 | 341 | 42.88% | 42.08% | 42.71% | 7.12 pp | -85 | 55 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 597 | 255 | 342 | 42.71% | 45.00% | 43.54% | 7.29 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 597 | 254 | 343 | 42.55% | 42.08% | 42.08% | 7.45 pp | -89 | 55 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 651 | 305 | 346 | 46.85% | 48.33% | 47.71% | 3.15 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 651 | 304 | 347 | 46.70% | 48.33% | 47.29% | 3.30 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 651 | 304 | 347 | 46.70% | 47.92% | 48.12% | 3.30 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 651 | 269 | 382 | 41.32% | 41.67% | 41.04% | 8.68 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 651 | 268 | 383 | 41.17% | 42.92% | 40.62% | 8.83 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 651 | 266 | 385 | 40.86% | 42.50% | 40.21% | 9.14 pp | -119 | 55 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 15 | -2.73 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 103 | 134 | 43.46% | 43.46% | 43.46% | 6.54 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 98 | 139 | 41.35% | 41.35% | 41.35% | 8.65 pp | -41 | 15 | -2.73 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 7 | -3.86 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
