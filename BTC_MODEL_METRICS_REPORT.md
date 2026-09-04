# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T11:44:27.767053+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1226 | 938 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1102 | 737 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 770 | 499 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 772 | 553 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 147 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 147 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 35 | 112 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 35 | 112 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 499 | 242 | 257 | 48.50% | 45.42% | 48.33% | 1.50 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 499 | 237 | 262 | 47.49% | 51.25% | 48.12% | 2.51 pp | -25 | 48 | -0.52 |
| BTC Daily | mlp_sklearn | MLPClassifier | 727 | 351 | 376 | 48.28% | 46.25% | 47.92% | 1.72 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 499 | 235 | 264 | 47.09% | 44.58% | 47.71% | 2.91 pp | -29 | 48 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| BTC Daily | transformer | Transformer | 727 | 347 | 380 | 47.73% | 47.08% | 49.79% | 2.27 pp | -33 | 43 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 553 | 258 | 295 | 46.65% | 49.17% | 47.50% | 3.35 pp | -37 | 48 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 904 | 433 | 471 | 47.90% | 51.25% | 48.54% | 2.10 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 553 | 256 | 297 | 46.29% | 49.58% | 46.88% | 3.71 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 553 | 256 | 297 | 46.29% | 45.00% | 47.71% | 3.71 pp | -41 | 48 | -0.85 |
| Consolidated Hourly | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 904 | 427 | 477 | 47.23% | 47.50% | 46.67% | 2.77 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 727 | 336 | 391 | 46.22% | 44.17% | 47.08% | 3.78 pp | -55 | 43 | -1.28 |
| BTC Market Hours | lstm | LSTM | 499 | 216 | 283 | 43.29% | 41.67% | 43.33% | 6.71 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 499 | 216 | 283 | 43.29% | 44.58% | 43.54% | 6.71 pp | -67 | 48 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 499 | 205 | 294 | 41.08% | 41.67% | 41.25% | 8.92 pp | -89 | 48 | -1.85 |
| Consolidated Hourly | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 553 | 230 | 323 | 41.59% | 42.50% | 40.83% | 8.41 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 904 | 402 | 502 | 44.47% | 44.17% | 42.08% | 5.53 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 904 | 401 | 503 | 44.36% | 43.75% | 43.75% | 5.64 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 553 | 223 | 330 | 40.33% | 38.75% | 40.42% | 9.67 pp | -107 | 48 | -2.23 |
| Consolidated Hourly | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 553 | 221 | 332 | 39.96% | 41.25% | 38.96% | 10.04 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 727 | 313 | 414 | 43.05% | 37.08% | 41.46% | 6.95 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 727 | 311 | 416 | 42.78% | 40.83% | 43.54% | 7.22 pp | -105 | 43 | -2.44 |
| BTC Hourly | lstm | LSTM | 904 | 387 | 517 | 42.81% | 39.58% | 41.88% | 7.19 pp | -130 | 48 | -2.71 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 904 | 379 | 525 | 41.92% | 41.25% | 41.04% | 8.08 pp | -146 | 48 | -3.04 |
| BTC Daily | xgb | XGBoost | 737 | 293 | 444 | 39.76% | 37.50% | 38.54% | 10.24 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 904 | 433 | 471 | 47.90% | 51.25% | 48.54% | 2.10 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 904 | 427 | 477 | 47.23% | 47.50% | 46.67% | 2.77 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 904 | 402 | 502 | 44.47% | 44.17% | 42.08% | 5.53 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 904 | 401 | 503 | 44.36% | 43.75% | 43.75% | 5.64 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 904 | 387 | 517 | 42.81% | 39.58% | 41.88% | 7.19 pp | -130 | 48 | -2.71 |
| BTC Hourly | xgb | XGBoost | 904 | 379 | 525 | 41.92% | 41.25% | 41.04% | 8.08 pp | -146 | 48 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 727 | 351 | 376 | 48.28% | 46.25% | 47.92% | 1.72 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 727 | 347 | 380 | 47.73% | 47.08% | 49.79% | 2.27 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 727 | 336 | 391 | 46.22% | 44.17% | 47.08% | 3.78 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 727 | 313 | 414 | 43.05% | 37.08% | 41.46% | 6.95 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 727 | 311 | 416 | 42.78% | 40.83% | 43.54% | 7.22 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 737 | 293 | 444 | 39.76% | 37.50% | 38.54% | 10.24 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 499 | 242 | 257 | 48.50% | 45.42% | 48.33% | 1.50 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 499 | 237 | 262 | 47.49% | 51.25% | 48.12% | 2.51 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 499 | 235 | 264 | 47.09% | 44.58% | 47.71% | 2.91 pp | -29 | 48 | -0.60 |
| BTC Market Hours | lstm | LSTM | 499 | 216 | 283 | 43.29% | 41.67% | 43.33% | 6.71 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 499 | 216 | 283 | 43.29% | 44.58% | 43.54% | 6.71 pp | -67 | 48 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 499 | 205 | 294 | 41.08% | 41.67% | 41.25% | 8.92 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 553 | 258 | 295 | 46.65% | 49.17% | 47.50% | 3.35 pp | -37 | 48 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 553 | 256 | 297 | 46.29% | 49.58% | 46.88% | 3.71 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 553 | 256 | 297 | 46.29% | 45.00% | 47.71% | 3.71 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 553 | 230 | 323 | 41.59% | 42.50% | 40.83% | 8.41 pp | -93 | 48 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 553 | 223 | 330 | 40.33% | 38.75% | 40.42% | 9.67 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 553 | 221 | 332 | 39.96% | 41.25% | 38.96% | 10.04 pp | -111 | 48 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
