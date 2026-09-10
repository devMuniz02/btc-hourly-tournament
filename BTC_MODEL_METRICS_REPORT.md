# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T07:36:46.764872+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1321 | 1033 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1196 | 831 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 942 | 593 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 944 | 647 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 593 | 287 | 306 | 48.40% | 47.08% | 47.29% | 1.60 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 593 | 287 | 306 | 48.40% | 52.50% | 50.21% | 1.60 pp | -19 | 55 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| BTC Market Hours | transformer | Transformer | 593 | 279 | 314 | 47.05% | 46.67% | 46.46% | 2.95 pp | -35 | 55 | -0.64 |
| BTC Market Hours Daily | nn | NN | 647 | 304 | 343 | 46.99% | 48.75% | 48.12% | 3.01 pp | -39 | 55 | -0.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 821 | 393 | 428 | 47.87% | 44.58% | 46.88% | 2.13 pp | -35 | 47 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 647 | 303 | 344 | 46.83% | 48.75% | 47.29% | 3.17 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 647 | 302 | 345 | 46.68% | 48.33% | 47.92% | 3.32 pp | -43 | 55 | -0.78 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 999 | 473 | 526 | 47.35% | 48.75% | 46.04% | 2.65 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 821 | 382 | 439 | 46.53% | 45.00% | 45.21% | 3.47 pp | -57 | 47 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 999 | 465 | 534 | 46.55% | 45.83% | 45.00% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Daily | transformer | Transformer | 821 | 379 | 442 | 46.16% | 38.33% | 45.21% | 3.84 pp | -63 | 47 | -1.34 |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 593 | 255 | 338 | 43.00% | 45.42% | 43.54% | 7.00 pp | -83 | 55 | -1.51 |
| BTC Market Hours | lstm | LSTM | 593 | 253 | 340 | 42.66% | 42.50% | 42.50% | 7.34 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 593 | 253 | 340 | 42.66% | 42.50% | 42.50% | 7.34 pp | -87 | 55 | -1.58 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 647 | 267 | 380 | 41.27% | 41.25% | 40.62% | 8.73 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 647 | 267 | 380 | 41.27% | 43.33% | 40.42% | 8.73 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 647 | 264 | 383 | 40.80% | 42.08% | 40.21% | 9.20 pp | -119 | 55 | -2.16 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Hourly | nn | NN | 999 | 440 | 559 | 44.04% | 42.08% | 41.67% | 5.96 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 999 | 440 | 559 | 44.04% | 41.67% | 43.12% | 5.96 pp | -119 | 52 | -2.29 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| BTC Daily | lstm | LSTM | 821 | 348 | 473 | 42.39% | 35.83% | 41.04% | 7.61 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 999 | 423 | 576 | 42.34% | 37.08% | 40.00% | 7.66 pp | -153 | 52 | -2.94 |
| BTC Daily | rf | RandomForest | 821 | 341 | 480 | 41.53% | 37.50% | 41.04% | 8.47 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| BTC Hourly | xgb | XGBoost | 999 | 411 | 588 | 41.14% | 35.42% | 38.54% | 8.86 pp | -177 | 52 | -3.40 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 831 | 327 | 504 | 39.35% | 37.50% | 36.25% | 10.65 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 999 | 473 | 526 | 47.35% | 48.75% | 46.04% | 2.65 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 999 | 465 | 534 | 46.55% | 45.83% | 45.00% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 999 | 440 | 559 | 44.04% | 42.08% | 41.67% | 5.96 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 999 | 440 | 559 | 44.04% | 41.67% | 43.12% | 5.96 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 999 | 423 | 576 | 42.34% | 37.08% | 40.00% | 7.66 pp | -153 | 52 | -2.94 |
| BTC Hourly | xgb | XGBoost | 999 | 411 | 588 | 41.14% | 35.42% | 38.54% | 8.86 pp | -177 | 52 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 821 | 393 | 428 | 47.87% | 44.58% | 46.88% | 2.13 pp | -35 | 47 | -0.74 |
| BTC Daily | nn | NN | 821 | 382 | 439 | 46.53% | 45.00% | 45.21% | 3.47 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 821 | 379 | 442 | 46.16% | 38.33% | 45.21% | 3.84 pp | -63 | 47 | -1.34 |
| BTC Daily | lstm | LSTM | 821 | 348 | 473 | 42.39% | 35.83% | 41.04% | 7.61 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 821 | 341 | 480 | 41.53% | 37.50% | 41.04% | 8.47 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 831 | 327 | 504 | 39.35% | 37.50% | 36.25% | 10.65 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 593 | 287 | 306 | 48.40% | 47.08% | 47.29% | 1.60 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 593 | 287 | 306 | 48.40% | 52.50% | 50.21% | 1.60 pp | -19 | 55 | -0.35 |
| BTC Market Hours | transformer | Transformer | 593 | 279 | 314 | 47.05% | 46.67% | 46.46% | 2.95 pp | -35 | 55 | -0.64 |
| BTC Market Hours | xgb | XGBoost | 593 | 255 | 338 | 43.00% | 45.42% | 43.54% | 7.00 pp | -83 | 55 | -1.51 |
| BTC Market Hours | lstm | LSTM | 593 | 253 | 340 | 42.66% | 42.50% | 42.50% | 7.34 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 593 | 253 | 340 | 42.66% | 42.50% | 42.50% | 7.34 pp | -87 | 55 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 647 | 304 | 343 | 46.99% | 48.75% | 48.12% | 3.01 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 647 | 303 | 344 | 46.83% | 48.75% | 47.29% | 3.17 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 647 | 302 | 345 | 46.68% | 48.33% | 47.92% | 3.32 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 647 | 267 | 380 | 41.27% | 41.25% | 40.62% | 8.73 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 647 | 267 | 380 | 41.27% | 43.33% | 40.42% | 8.73 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 647 | 264 | 383 | 40.80% | 42.08% | 40.21% | 9.20 pp | -119 | 55 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
