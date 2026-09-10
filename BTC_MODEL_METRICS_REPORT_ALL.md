# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T09:58:44.571266+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1322 | 1034 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1198 | 833 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 944 | 595 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 946 | 649 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T15:00:00+00:00 | 235 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T15:00:00+00:00 | 235 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T15:00:00+00:00 | 235 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T15:00:00+00:00 | 236 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 595 | 287 | 308 | 48.24% | 46.25% | 47.08% | 1.76 pp | -21 | 55 | -0.38 |
| BTC Market Hours | nn | NN | 595 | 287 | 308 | 48.24% | 51.67% | 50.00% | 1.76 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 595 | 279 | 316 | 46.89% | 46.25% | 46.25% | 3.11 pp | -37 | 55 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 823 | 395 | 428 | 48.00% | 45.00% | 46.88% | 2.00 pp | -33 | 47 | -0.70 |
| BTC Market Hours Daily | nn | NN | 649 | 305 | 344 | 47.00% | 48.75% | 48.12% | 3.00 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 649 | 304 | 345 | 46.84% | 48.33% | 47.29% | 3.16 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 649 | 303 | 346 | 46.69% | 48.33% | 47.92% | 3.31 pp | -43 | 55 | -0.78 |
| Consolidated Hourly | rf | RandomForest | 235 | 111 | 124 | 47.23% | 47.23% | 47.23% | 2.77 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 111 | 124 | 47.23% | 47.23% | 47.23% | 2.77 pp | -13 | 15 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1000 | 473 | 527 | 47.30% | 48.33% | 45.83% | 2.70 pp | -54 | 52 | -1.04 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 823 | 384 | 439 | 46.66% | 45.42% | 45.42% | 3.34 pp | -55 | 47 | -1.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 823 | 381 | 442 | 46.29% | 38.33% | 45.21% | 3.71 pp | -61 | 47 | -1.30 |
| BTC Hourly | transformer | Transformer | 1000 | 466 | 534 | 46.60% | 46.25% | 45.00% | 3.40 pp | -68 | 52 | -1.31 |
| Consolidated Hourly | lstm | LSTM | 235 | 107 | 128 | 45.53% | 45.53% | 45.53% | 4.47 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 107 | 128 | 45.53% | 45.53% | 45.53% | 4.47 pp | -21 | 15 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 595 | 255 | 340 | 42.86% | 45.00% | 43.54% | 7.14 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 595 | 254 | 341 | 42.69% | 42.08% | 42.50% | 7.31 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 595 | 254 | 341 | 42.69% | 42.50% | 42.29% | 7.31 pp | -87 | 55 | -1.58 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | xgb | XGBoost | 235 | 103 | 132 | 43.83% | 43.83% | 43.83% | 6.17 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 103 | 132 | 43.83% | 43.83% | 43.83% | 6.17 pp | -29 | 15 | -1.93 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 649 | 268 | 381 | 41.29% | 41.25% | 40.83% | 8.71 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 649 | 268 | 381 | 41.29% | 42.92% | 40.62% | 8.71 pp | -113 | 55 | -2.05 |
| Consolidated Hourly | transformer | Transformer | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 649 | 265 | 384 | 40.83% | 42.08% | 40.21% | 9.17 pp | -119 | 55 | -2.16 |
| BTC Hourly | nn | NN | 1000 | 441 | 559 | 44.10% | 42.08% | 41.67% | 5.90 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1000 | 440 | 560 | 44.00% | 41.67% | 43.12% | 6.00 pp | -120 | 52 | -2.31 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 823 | 348 | 475 | 42.28% | 35.42% | 40.62% | 7.72 pp | -127 | 47 | -2.70 |
| Consolidated Hourly | nn | NN | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 15 | -2.73 |
| BTC Daily | rf | RandomForest | 823 | 343 | 480 | 41.68% | 37.50% | 41.04% | 8.32 pp | -137 | 47 | -2.91 |
| BTC Hourly | lstm | LSTM | 1000 | 423 | 577 | 42.30% | 37.08% | 39.79% | 7.70 pp | -154 | 52 | -2.96 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1000 | 412 | 588 | 41.20% | 35.83% | 38.75% | 8.80 pp | -176 | 52 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 833 | 329 | 504 | 39.50% | 37.50% | 36.67% | 10.50 pp | -175 | 47 | -3.72 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1000 | 473 | 527 | 47.30% | 48.33% | 45.83% | 2.70 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1000 | 466 | 534 | 46.60% | 46.25% | 45.00% | 3.40 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1000 | 441 | 559 | 44.10% | 42.08% | 41.67% | 5.90 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1000 | 440 | 560 | 44.00% | 41.67% | 43.12% | 6.00 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 1000 | 423 | 577 | 42.30% | 37.08% | 39.79% | 7.70 pp | -154 | 52 | -2.96 |
| BTC Hourly | xgb | XGBoost | 1000 | 412 | 588 | 41.20% | 35.83% | 38.75% | 8.80 pp | -176 | 52 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 823 | 395 | 428 | 48.00% | 45.00% | 46.88% | 2.00 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 823 | 384 | 439 | 46.66% | 45.42% | 45.42% | 3.34 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 823 | 381 | 442 | 46.29% | 38.33% | 45.21% | 3.71 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 823 | 348 | 475 | 42.28% | 35.42% | 40.62% | 7.72 pp | -127 | 47 | -2.70 |
| BTC Daily | rf | RandomForest | 823 | 343 | 480 | 41.68% | 37.50% | 41.04% | 8.32 pp | -137 | 47 | -2.91 |
| BTC Daily | xgb | XGBoost | 833 | 329 | 504 | 39.50% | 37.50% | 36.67% | 10.50 pp | -175 | 47 | -3.72 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 595 | 287 | 308 | 48.24% | 46.25% | 47.08% | 1.76 pp | -21 | 55 | -0.38 |
| BTC Market Hours | nn | NN | 595 | 287 | 308 | 48.24% | 51.67% | 50.00% | 1.76 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 595 | 279 | 316 | 46.89% | 46.25% | 46.25% | 3.11 pp | -37 | 55 | -0.67 |
| BTC Market Hours | xgb | XGBoost | 595 | 255 | 340 | 42.86% | 45.00% | 43.54% | 7.14 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 595 | 254 | 341 | 42.69% | 42.08% | 42.50% | 7.31 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 595 | 254 | 341 | 42.69% | 42.50% | 42.29% | 7.31 pp | -87 | 55 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 649 | 305 | 344 | 47.00% | 48.75% | 48.12% | 3.00 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 649 | 304 | 345 | 46.84% | 48.33% | 47.29% | 3.16 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 649 | 303 | 346 | 46.69% | 48.33% | 47.92% | 3.31 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 649 | 268 | 381 | 41.29% | 41.25% | 40.83% | 8.71 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 649 | 268 | 381 | 41.29% | 42.92% | 40.62% | 8.71 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 649 | 265 | 384 | 40.83% | 42.08% | 40.21% | 9.17 pp | -119 | 55 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 235 | 111 | 124 | 47.23% | 47.23% | 47.23% | 2.77 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 235 | 107 | 128 | 45.53% | 45.53% | 45.53% | 4.47 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 235 | 103 | 132 | 43.83% | 43.83% | 43.83% | 6.17 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | transformer | Transformer | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 15 | -2.73 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 111 | 124 | 47.23% | 47.23% | 47.23% | 2.77 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 107 | 128 | 45.53% | 45.53% | 45.53% | 4.47 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 103 | 132 | 43.83% | 43.83% | 43.83% | 6.17 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 15 | -2.73 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
