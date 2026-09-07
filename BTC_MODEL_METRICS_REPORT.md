# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T01:37:40.665277+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1144 | 779 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 851 | 541 | 309 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 541 | 262 | 279 | 48.43% | 46.25% | 47.92% | 1.57 pp | -17 | 51 | -0.33 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 541 | 258 | 283 | 47.69% | 48.75% | 48.12% | 2.31 pp | -25 | 51 | -0.49 |
| BTC Daily | mlp_sklearn | MLPClassifier | 769 | 371 | 398 | 48.24% | 46.25% | 48.12% | 1.76 pp | -27 | 45 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 595 | 282 | 313 | 47.39% | 50.83% | 48.75% | 2.61 pp | -31 | 51 | -0.61 |
| BTC Market Hours | nn | NN | 541 | 255 | 286 | 47.13% | 49.58% | 48.96% | 2.87 pp | -31 | 51 | -0.61 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 947 | 453 | 494 | 47.84% | 50.00% | 47.29% | 2.16 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | nn | NN | 595 | 276 | 319 | 46.39% | 46.25% | 47.92% | 3.61 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 595 | 275 | 320 | 46.22% | 50.00% | 46.88% | 3.78 pp | -45 | 51 | -0.88 |
| BTC Daily | transformer | Transformer | 769 | 359 | 410 | 46.68% | 41.25% | 46.88% | 3.32 pp | -51 | 45 | -1.13 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 769 | 356 | 413 | 46.29% | 44.58% | 45.83% | 3.71 pp | -57 | 45 | -1.27 |
| BTC Hourly | transformer | Transformer | 947 | 442 | 505 | 46.67% | 45.83% | 44.79% | 3.33 pp | -63 | 49 | -1.29 |
| BTC Market Hours | rf | RandomForest | 541 | 233 | 308 | 43.07% | 44.58% | 43.33% | 6.93 pp | -75 | 51 | -1.47 |
| BTC Market Hours | lstm | LSTM | 541 | 232 | 309 | 42.88% | 41.25% | 43.75% | 7.12 pp | -77 | 51 | -1.51 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 541 | 224 | 317 | 41.40% | 42.92% | 41.88% | 8.60 pp | -93 | 51 | -1.82 |
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
| BTC Daily | lstm | LSTM | 769 | 325 | 444 | 42.26% | 35.83% | 40.21% | 7.74 pp | -119 | 45 | -2.64 |
| BTC Daily | rf | RandomForest | 769 | 322 | 447 | 41.87% | 38.33% | 42.08% | 8.13 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 947 | 405 | 542 | 42.77% | 36.67% | 42.29% | 7.23 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | xgb | XGBoost | 947 | 396 | 551 | 41.82% | 40.00% | 40.42% | 8.18 pp | -155 | 49 | -3.16 |
| BTC Daily | xgb | XGBoost | 779 | 304 | 475 | 39.02% | 34.58% | 36.25% | 10.98 pp | -171 | 45 | -3.80 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 769 | 371 | 398 | 48.24% | 46.25% | 48.12% | 1.76 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 769 | 359 | 410 | 46.68% | 41.25% | 46.88% | 3.32 pp | -51 | 45 | -1.13 |
| BTC Daily | nn | NN | 769 | 356 | 413 | 46.29% | 44.58% | 45.83% | 3.71 pp | -57 | 45 | -1.27 |
| BTC Daily | lstm | LSTM | 769 | 325 | 444 | 42.26% | 35.83% | 40.21% | 7.74 pp | -119 | 45 | -2.64 |
| BTC Daily | rf | RandomForest | 769 | 322 | 447 | 41.87% | 38.33% | 42.08% | 8.13 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 779 | 304 | 475 | 39.02% | 34.58% | 36.25% | 10.98 pp | -171 | 45 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 541 | 262 | 279 | 48.43% | 46.25% | 47.92% | 1.57 pp | -17 | 51 | -0.33 |
| BTC Market Hours | transformer | Transformer | 541 | 258 | 283 | 47.69% | 48.75% | 48.12% | 2.31 pp | -25 | 51 | -0.49 |
| BTC Market Hours | nn | NN | 541 | 255 | 286 | 47.13% | 49.58% | 48.96% | 2.87 pp | -31 | 51 | -0.61 |
| BTC Market Hours | rf | RandomForest | 541 | 233 | 308 | 43.07% | 44.58% | 43.33% | 6.93 pp | -75 | 51 | -1.47 |
| BTC Market Hours | lstm | LSTM | 541 | 232 | 309 | 42.88% | 41.25% | 43.75% | 7.12 pp | -77 | 51 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 541 | 224 | 317 | 41.40% | 42.92% | 41.88% | 8.60 pp | -93 | 51 | -1.82 |

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
