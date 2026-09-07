# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T04:52:33.607690+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 856 | 598 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 188 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 188 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 188 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 189 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 544 | 275 | 269 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 188 | 94 | 94 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 188 | 94 | 94 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | nn | NN | 544 | 269 | 275 | 49.45% | 51.67% | 51.04% | 0.55 pp | -6 | 51 | -0.12 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 188 | 92 | 96 | 48.94% | 48.94% | 48.94% | 1.06 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 188 | 92 | 96 | 48.94% | 48.94% | 48.94% | 1.06 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | nn | NN | 598 | 286 | 312 | 47.83% | 47.50% | 48.12% | 2.17 pp | -26 | 51 | -0.51 |
| BTC Daily | mlp_sklearn | MLPClassifier | 772 | 373 | 399 | 48.32% | 46.67% | 48.12% | 1.68 pp | -26 | 45 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 598 | 281 | 317 | 46.99% | 47.92% | 46.67% | 3.01 pp | -36 | 51 | -0.71 |
| BTC Market Hours | transformer | Transformer | 544 | 253 | 291 | 46.51% | 46.25% | 47.29% | 3.49 pp | -38 | 51 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 949 | 453 | 496 | 47.73% | 49.58% | 47.08% | 2.27 pp | -43 | 50 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 598 | 277 | 321 | 46.32% | 47.92% | 47.71% | 3.68 pp | -44 | 51 | -0.86 |
| Consolidated Hourly | xgb | XGBoost | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 13 | -1.08 |
| BTC Daily | transformer | Transformer | 772 | 360 | 412 | 46.63% | 41.25% | 46.88% | 3.37 pp | -52 | 45 | -1.16 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| BTC Market Hours | rf | RandomForest | 544 | 241 | 303 | 44.30% | 46.67% | 44.17% | 5.70 pp | -62 | 51 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 188 | 86 | 102 | 45.74% | 45.74% | 45.74% | 4.26 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 188 | 86 | 102 | 45.74% | 45.74% | 45.74% | 4.26 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 772 | 357 | 415 | 46.24% | 44.17% | 45.62% | 3.76 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 949 | 442 | 507 | 46.58% | 45.42% | 44.38% | 3.42 pp | -65 | 50 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 598 | 264 | 334 | 44.15% | 46.25% | 43.54% | 5.85 pp | -70 | 51 | -1.37 |
| Consolidated Hourly | nn | NN | 188 | 85 | 103 | 45.21% | 45.21% | 45.21% | 4.79 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 188 | 85 | 103 | 45.21% | 45.21% | 45.21% | 4.79 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 544 | 226 | 318 | 41.54% | 36.67% | 41.67% | 8.46 pp | -92 | 51 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 544 | 225 | 319 | 41.36% | 42.50% | 41.67% | 8.64 pp | -94 | 51 | -1.84 |
| BTC Market Hours Daily | xgb | XGBoost | 598 | 251 | 347 | 41.97% | 42.92% | 41.46% | 8.03 pp | -96 | 51 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Hourly | rf | RandomForest | 949 | 422 | 527 | 44.47% | 44.58% | 43.75% | 5.53 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 949 | 420 | 529 | 44.26% | 42.08% | 42.71% | 5.74 pp | -109 | 50 | -2.18 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 598 | 237 | 361 | 39.63% | 35.83% | 39.17% | 10.37 pp | -124 | 51 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
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
| BTC Market Hours Daily | nn | NN | 598 | 286 | 312 | 47.83% | 47.50% | 48.12% | 2.17 pp | -26 | 51 | -0.51 |
| BTC Market Hours Daily | transformer | Transformer | 598 | 281 | 317 | 46.99% | 47.92% | 46.67% | 3.01 pp | -36 | 51 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 598 | 277 | 321 | 46.32% | 47.92% | 47.71% | 3.68 pp | -44 | 51 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 598 | 264 | 334 | 44.15% | 46.25% | 43.54% | 5.85 pp | -70 | 51 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 598 | 251 | 347 | 41.97% | 42.92% | 41.46% | 8.03 pp | -96 | 51 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 598 | 237 | 361 | 39.63% | 35.83% | 39.17% | 10.37 pp | -124 | 51 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 188 | 94 | 94 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 188 | 92 | 96 | 48.94% | 48.94% | 48.94% | 1.06 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 188 | 86 | 102 | 45.74% | 45.74% | 45.74% | 4.26 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | nn | NN | 188 | 85 | 103 | 45.21% | 45.21% | 45.21% | 4.79 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 188 | 94 | 94 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 188 | 92 | 96 | 48.94% | 48.94% | 48.94% | 1.06 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 188 | 86 | 102 | 45.74% | 45.74% | 45.74% | 4.26 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | nn | NN | 188 | 85 | 103 | 45.21% | 45.21% | 45.21% | 4.79 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 188 | 81 | 107 | 43.09% | 43.09% | 43.09% | 6.91 pp | -26 | 13 | -2.00 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
