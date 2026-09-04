# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T17:20:30.362857+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1230 | 942 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1106 | 741 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 16:00:00+00:00 | 779 | 503 | 275 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 16:00:00+00:00 | 781 | 557 | 222 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 151 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 151 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 151 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 152 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 151 | 75 | 76 | 49.67% | 49.67% | 49.67% | 0.33 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 75 | 76 | 49.67% | 49.67% | 49.67% | 0.33 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 503 | 242 | 261 | 48.11% | 44.58% | 48.12% | 1.89 pp | -19 | 48 | -0.40 |
| BTC Market Hours | nn | NN | 503 | 239 | 264 | 47.51% | 50.83% | 48.33% | 2.49 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 503 | 239 | 264 | 47.51% | 45.83% | 48.12% | 2.49 pp | -25 | 48 | -0.52 |
| BTC Daily | mlp_sklearn | MLPClassifier | 731 | 353 | 378 | 48.29% | 46.67% | 47.92% | 1.71 pp | -25 | 43 | -0.58 |
| Consolidated Hourly | xgb | XGBoost | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 11 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 557 | 262 | 295 | 47.04% | 49.58% | 47.92% | 2.96 pp | -33 | 48 | -0.69 |
| BTC Daily | transformer | Transformer | 731 | 349 | 382 | 47.74% | 47.08% | 49.79% | 2.26 pp | -33 | 43 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 908 | 435 | 473 | 47.91% | 51.25% | 48.33% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 557 | 258 | 299 | 46.32% | 50.00% | 46.67% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 557 | 258 | 299 | 46.32% | 45.42% | 47.50% | 3.68 pp | -41 | 48 | -0.85 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 908 | 429 | 479 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 731 | 338 | 393 | 46.24% | 45.00% | 46.88% | 3.76 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 503 | 217 | 286 | 43.14% | 41.67% | 43.33% | 6.86 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 503 | 216 | 287 | 42.94% | 44.17% | 43.12% | 7.06 pp | -71 | 48 | -1.48 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 503 | 207 | 296 | 41.15% | 42.08% | 41.67% | 8.85 pp | -89 | 48 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 557 | 231 | 326 | 41.47% | 42.08% | 40.42% | 8.53 pp | -95 | 48 | -1.98 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 908 | 404 | 504 | 44.49% | 44.17% | 42.08% | 5.51 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 908 | 403 | 505 | 44.38% | 44.17% | 43.75% | 5.62 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 557 | 225 | 332 | 40.39% | 38.75% | 40.42% | 9.61 pp | -107 | 48 | -2.23 |
| Consolidated Hourly | transformer | Transformer | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 11 | -2.27 |
| BTC Daily | lstm | LSTM | 731 | 315 | 416 | 43.09% | 37.08% | 41.46% | 6.91 pp | -101 | 43 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 557 | 222 | 335 | 39.86% | 40.42% | 38.75% | 10.14 pp | -113 | 48 | -2.35 |
| BTC Daily | rf | RandomForest | 731 | 312 | 419 | 42.68% | 41.25% | 43.33% | 7.32 pp | -107 | 43 | -2.49 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 908 | 389 | 519 | 42.84% | 40.00% | 42.08% | 7.16 pp | -130 | 48 | -2.71 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 908 | 381 | 527 | 41.96% | 41.25% | 40.83% | 8.04 pp | -146 | 48 | -3.04 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 741 | 293 | 448 | 39.54% | 36.25% | 38.12% | 10.46 pp | -155 | 43 | -3.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 908 | 435 | 473 | 47.91% | 51.25% | 48.33% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 908 | 429 | 479 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 908 | 404 | 504 | 44.49% | 44.17% | 42.08% | 5.51 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 908 | 403 | 505 | 44.38% | 44.17% | 43.75% | 5.62 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 908 | 389 | 519 | 42.84% | 40.00% | 42.08% | 7.16 pp | -130 | 48 | -2.71 |
| BTC Hourly | xgb | XGBoost | 908 | 381 | 527 | 41.96% | 41.25% | 40.83% | 8.04 pp | -146 | 48 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 731 | 353 | 378 | 48.29% | 46.67% | 47.92% | 1.71 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 731 | 349 | 382 | 47.74% | 47.08% | 49.79% | 2.26 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 731 | 338 | 393 | 46.24% | 45.00% | 46.88% | 3.76 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 731 | 315 | 416 | 43.09% | 37.08% | 41.46% | 6.91 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 731 | 312 | 419 | 42.68% | 41.25% | 43.33% | 7.32 pp | -107 | 43 | -2.49 |
| BTC Daily | xgb | XGBoost | 741 | 293 | 448 | 39.54% | 36.25% | 38.12% | 10.46 pp | -155 | 43 | -3.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 503 | 242 | 261 | 48.11% | 44.58% | 48.12% | 1.89 pp | -19 | 48 | -0.40 |
| BTC Market Hours | nn | NN | 503 | 239 | 264 | 47.51% | 50.83% | 48.33% | 2.49 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 503 | 239 | 264 | 47.51% | 45.83% | 48.12% | 2.49 pp | -25 | 48 | -0.52 |
| BTC Market Hours | lstm | LSTM | 503 | 217 | 286 | 43.14% | 41.67% | 43.33% | 6.86 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 503 | 216 | 287 | 42.94% | 44.17% | 43.12% | 7.06 pp | -71 | 48 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 503 | 207 | 296 | 41.15% | 42.08% | 41.67% | 8.85 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 557 | 262 | 295 | 47.04% | 49.58% | 47.92% | 2.96 pp | -33 | 48 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 557 | 258 | 299 | 46.32% | 50.00% | 46.67% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 557 | 258 | 299 | 46.32% | 45.42% | 47.50% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 557 | 231 | 326 | 41.47% | 42.08% | 40.42% | 8.53 pp | -95 | 48 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 557 | 225 | 332 | 40.39% | 38.75% | 40.42% | 9.61 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 557 | 222 | 335 | 39.86% | 40.42% | 38.75% | 10.14 pp | -113 | 48 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 151 | 75 | 76 | 49.67% | 49.67% | 49.67% | 0.33 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 75 | 76 | 49.67% | 49.67% | 49.67% | 0.33 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
