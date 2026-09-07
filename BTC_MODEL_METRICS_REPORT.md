# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T01:45:27.248682+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1269 | 981 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1145 | 780 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 852 | 542 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 853 | 595 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 263 | 279 | 48.52% | 46.25% | 47.92% | 1.48 pp | -16 | 51 | -0.31 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 542 | 259 | 283 | 47.79% | 48.75% | 48.12% | 2.21 pp | -24 | 51 | -0.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 770 | 372 | 398 | 48.31% | 46.67% | 48.12% | 1.69 pp | -26 | 45 | -0.58 |
| BTC Market Hours | nn | NN | 542 | 256 | 286 | 47.23% | 50.00% | 48.96% | 2.77 pp | -30 | 51 | -0.59 |
| BTC Market Hours Daily | transformer | Transformer | 595 | 282 | 313 | 47.39% | 50.83% | 48.75% | 2.61 pp | -31 | 51 | -0.61 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 947 | 453 | 494 | 47.84% | 50.00% | 47.29% | 2.16 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | nn | NN | 595 | 276 | 319 | 46.39% | 46.25% | 47.92% | 3.61 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 595 | 275 | 320 | 46.22% | 50.00% | 46.88% | 3.78 pp | -45 | 51 | -0.88 |
| BTC Daily | transformer | Transformer | 770 | 360 | 410 | 46.75% | 41.67% | 47.08% | 3.25 pp | -50 | 45 | -1.11 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 770 | 357 | 413 | 46.36% | 45.00% | 46.04% | 3.64 pp | -56 | 45 | -1.24 |
| BTC Hourly | transformer | Transformer | 947 | 442 | 505 | 46.67% | 45.83% | 44.79% | 3.33 pp | -63 | 49 | -1.29 |
| BTC Market Hours | rf | RandomForest | 542 | 234 | 308 | 43.17% | 45.00% | 43.33% | 6.83 pp | -74 | 51 | -1.45 |
| BTC Market Hours | lstm | LSTM | 542 | 233 | 309 | 42.99% | 41.67% | 43.75% | 7.01 pp | -76 | 51 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 542 | 225 | 317 | 41.51% | 43.33% | 41.88% | 8.49 pp | -92 | 51 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 595 | 248 | 347 | 41.68% | 45.00% | 41.25% | 8.32 pp | -99 | 51 | -1.94 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 947 | 420 | 527 | 44.35% | 42.08% | 42.92% | 5.65 pp | -107 | 49 | -2.18 |
| BTC Hourly | rf | RandomForest | 947 | 420 | 527 | 44.35% | 44.17% | 43.75% | 5.65 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 595 | 240 | 355 | 40.34% | 38.75% | 40.00% | 9.66 pp | -115 | 51 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 595 | 237 | 358 | 39.83% | 41.25% | 38.96% | 10.17 pp | -121 | 51 | -2.37 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 770 | 325 | 445 | 42.21% | 35.83% | 40.21% | 7.79 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 770 | 323 | 447 | 41.95% | 38.75% | 42.29% | 8.05 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 947 | 405 | 542 | 42.77% | 36.67% | 42.29% | 7.23 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | xgb | XGBoost | 947 | 396 | 551 | 41.82% | 40.00% | 40.42% | 8.18 pp | -155 | 49 | -3.16 |
| BTC Daily | xgb | XGBoost | 780 | 305 | 475 | 39.10% | 35.00% | 36.46% | 10.90 pp | -170 | 45 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 947 | 453 | 494 | 47.84% | 50.00% | 47.29% | 2.16 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 947 | 442 | 505 | 46.67% | 45.83% | 44.79% | 3.33 pp | -63 | 49 | -1.29 |
| BTC Hourly | nn | NN | 947 | 420 | 527 | 44.35% | 42.08% | 42.92% | 5.65 pp | -107 | 49 | -2.18 |
| BTC Hourly | rf | RandomForest | 947 | 420 | 527 | 44.35% | 44.17% | 43.75% | 5.65 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 947 | 405 | 542 | 42.77% | 36.67% | 42.29% | 7.23 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 947 | 396 | 551 | 41.82% | 40.00% | 40.42% | 8.18 pp | -155 | 49 | -3.16 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 770 | 372 | 398 | 48.31% | 46.67% | 48.12% | 1.69 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 770 | 360 | 410 | 46.75% | 41.67% | 47.08% | 3.25 pp | -50 | 45 | -1.11 |
| BTC Daily | nn | NN | 770 | 357 | 413 | 46.36% | 45.00% | 46.04% | 3.64 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 770 | 325 | 445 | 42.21% | 35.83% | 40.21% | 7.79 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 770 | 323 | 447 | 41.95% | 38.75% | 42.29% | 8.05 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 780 | 305 | 475 | 39.10% | 35.00% | 36.46% | 10.90 pp | -170 | 45 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 263 | 279 | 48.52% | 46.25% | 47.92% | 1.48 pp | -16 | 51 | -0.31 |
| BTC Market Hours | transformer | Transformer | 542 | 259 | 283 | 47.79% | 48.75% | 48.12% | 2.21 pp | -24 | 51 | -0.47 |
| BTC Market Hours | nn | NN | 542 | 256 | 286 | 47.23% | 50.00% | 48.96% | 2.77 pp | -30 | 51 | -0.59 |
| BTC Market Hours | rf | RandomForest | 542 | 234 | 308 | 43.17% | 45.00% | 43.33% | 6.83 pp | -74 | 51 | -1.45 |
| BTC Market Hours | lstm | LSTM | 542 | 233 | 309 | 42.99% | 41.67% | 43.75% | 7.01 pp | -76 | 51 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 542 | 225 | 317 | 41.51% | 43.33% | 41.88% | 8.49 pp | -92 | 51 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 595 | 282 | 313 | 47.39% | 50.83% | 48.75% | 2.61 pp | -31 | 51 | -0.61 |
| BTC Market Hours Daily | nn | NN | 595 | 276 | 319 | 46.39% | 46.25% | 47.92% | 3.61 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 595 | 275 | 320 | 46.22% | 50.00% | 46.88% | 3.78 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 595 | 248 | 347 | 41.68% | 45.00% | 41.25% | 8.32 pp | -99 | 51 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 595 | 240 | 355 | 40.34% | 38.75% | 40.00% | 9.66 pp | -115 | 51 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 595 | 237 | 358 | 39.83% | 41.25% | 38.96% | 10.17 pp | -121 | 51 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
