# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T02:04:27.858109+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 854 | 596 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 187 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 274 | 268 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| BTC Market Hours | nn | NN | 542 | 267 | 275 | 49.26% | 51.25% | 50.83% | 0.74 pp | -8 | 51 | -0.16 |
| Consolidated Hourly | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 596 | 284 | 312 | 47.65% | 46.67% | 48.12% | 2.35 pp | -28 | 51 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 770 | 372 | 398 | 48.31% | 46.67% | 48.12% | 1.69 pp | -26 | 45 | -0.58 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 596 | 280 | 316 | 46.98% | 47.92% | 46.88% | 3.02 pp | -36 | 51 | -0.71 |
| BTC Market Hours | transformer | Transformer | 542 | 252 | 290 | 46.49% | 45.83% | 47.08% | 3.51 pp | -38 | 51 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 947 | 453 | 494 | 47.84% | 50.00% | 47.29% | 2.16 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 596 | 275 | 321 | 46.14% | 47.92% | 47.29% | 3.86 pp | -46 | 51 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| BTC Daily | transformer | Transformer | 770 | 360 | 410 | 46.75% | 41.67% | 47.08% | 3.25 pp | -50 | 45 | -1.11 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 770 | 357 | 413 | 46.36% | 45.00% | 46.04% | 3.64 pp | -56 | 45 | -1.24 |
| BTC Market Hours | rf | RandomForest | 542 | 239 | 303 | 44.10% | 46.25% | 43.96% | 5.90 pp | -64 | 51 | -1.25 |
| BTC Hourly | transformer | Transformer | 947 | 442 | 505 | 46.67% | 45.83% | 44.79% | 3.33 pp | -63 | 49 | -1.29 |
| Consolidated Hourly | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 596 | 262 | 334 | 43.96% | 46.25% | 43.54% | 6.04 pp | -72 | 51 | -1.41 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 542 | 226 | 316 | 41.70% | 37.08% | 41.88% | 8.30 pp | -90 | 51 | -1.76 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 542 | 224 | 318 | 41.33% | 42.92% | 41.67% | 8.67 pp | -94 | 51 | -1.84 |
| BTC Market Hours Daily | xgb | XGBoost | 596 | 249 | 347 | 41.78% | 42.50% | 41.46% | 8.22 pp | -98 | 51 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Hourly | nn | NN | 947 | 420 | 527 | 44.35% | 42.08% | 42.92% | 5.65 pp | -107 | 49 | -2.18 |
| BTC Hourly | rf | RandomForest | 947 | 420 | 527 | 44.35% | 44.17% | 43.75% | 5.65 pp | -107 | 49 | -2.18 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 596 | 236 | 360 | 39.60% | 35.42% | 39.17% | 10.40 pp | -124 | 51 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 770 | 325 | 445 | 42.21% | 35.83% | 40.21% | 7.79 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 770 | 323 | 447 | 41.95% | 38.75% | 42.29% | 8.05 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 947 | 405 | 542 | 42.77% | 36.67% | 42.29% | 7.23 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 274 | 268 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| BTC Market Hours | nn | NN | 542 | 267 | 275 | 49.26% | 51.25% | 50.83% | 0.74 pp | -8 | 51 | -0.16 |
| BTC Market Hours | transformer | Transformer | 542 | 252 | 290 | 46.49% | 45.83% | 47.08% | 3.51 pp | -38 | 51 | -0.75 |
| BTC Market Hours | rf | RandomForest | 542 | 239 | 303 | 44.10% | 46.25% | 43.96% | 5.90 pp | -64 | 51 | -1.25 |
| BTC Market Hours | lstm | LSTM | 542 | 226 | 316 | 41.70% | 37.08% | 41.88% | 8.30 pp | -90 | 51 | -1.76 |
| BTC Market Hours | xgb | XGBoost | 542 | 224 | 318 | 41.33% | 42.92% | 41.67% | 8.67 pp | -94 | 51 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 596 | 284 | 312 | 47.65% | 46.67% | 48.12% | 2.35 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 596 | 280 | 316 | 46.98% | 47.92% | 46.88% | 3.02 pp | -36 | 51 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 596 | 275 | 321 | 46.14% | 47.92% | 47.29% | 3.86 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 596 | 262 | 334 | 43.96% | 46.25% | 43.54% | 6.04 pp | -72 | 51 | -1.41 |
| BTC Market Hours Daily | xgb | XGBoost | 596 | 249 | 347 | 41.78% | 42.50% | 41.46% | 8.22 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 596 | 236 | 360 | 39.60% | 35.42% | 39.17% | 10.40 pp | -124 | 51 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
