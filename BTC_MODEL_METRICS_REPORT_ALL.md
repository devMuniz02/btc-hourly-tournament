# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T17:32:15.402939+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1296 | 1008 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1172 | 807 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 897 | 569 | 327 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 898 | 622 | 274 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 569 | 277 | 292 | 48.68% | 47.50% | 48.12% | 1.32 pp | -15 | 53 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 569 | 270 | 299 | 47.45% | 51.67% | 49.17% | 2.55 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 797 | 385 | 412 | 48.31% | 47.08% | 47.92% | 1.69 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 569 | 267 | 302 | 46.92% | 46.67% | 46.88% | 3.08 pp | -35 | 53 | -0.66 |
| BTC Market Hours Daily | transformer | Transformer | 622 | 291 | 331 | 46.78% | 49.17% | 47.71% | 3.22 pp | -40 | 53 | -0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | nn | NN | 622 | 290 | 332 | 46.62% | 47.50% | 47.92% | 3.38 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 622 | 289 | 333 | 46.46% | 48.75% | 47.08% | 3.54 pp | -44 | 53 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 974 | 462 | 512 | 47.43% | 49.17% | 46.46% | 2.57 pp | -50 | 51 | -0.98 |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 797 | 370 | 427 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 797 | 370 | 427 | 46.42% | 40.00% | 46.25% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Hourly | transformer | Transformer | 974 | 453 | 521 | 46.51% | 44.58% | 43.75% | 3.49 pp | -68 | 51 | -1.33 |
| BTC Market Hours | lstm | LSTM | 569 | 247 | 322 | 43.41% | 42.50% | 43.96% | 6.59 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 569 | 245 | 324 | 43.06% | 45.42% | 43.75% | 6.94 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 569 | 244 | 325 | 42.88% | 46.25% | 43.33% | 7.12 pp | -81 | 53 | -1.53 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 622 | 260 | 362 | 41.80% | 43.75% | 40.83% | 8.20 pp | -102 | 53 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 622 | 257 | 365 | 41.32% | 44.17% | 40.83% | 8.68 pp | -108 | 53 | -2.04 |
| BTC Hourly | rf | RandomForest | 974 | 432 | 542 | 44.35% | 42.50% | 42.92% | 5.65 pp | -110 | 51 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 622 | 253 | 369 | 40.68% | 40.83% | 39.79% | 9.32 pp | -116 | 53 | -2.19 |
| BTC Hourly | nn | NN | 974 | 430 | 544 | 44.15% | 41.25% | 42.50% | 5.85 pp | -114 | 51 | -2.24 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 797 | 335 | 462 | 42.03% | 34.17% | 40.00% | 7.97 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 797 | 333 | 464 | 41.78% | 37.92% | 41.46% | 8.22 pp | -131 | 46 | -2.85 |
| BTC Hourly | lstm | LSTM | 974 | 414 | 560 | 42.51% | 37.92% | 41.04% | 7.49 pp | -146 | 51 | -2.86 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 974 | 402 | 572 | 41.27% | 35.83% | 38.96% | 8.73 pp | -170 | 51 | -3.33 |
| BTC Daily | xgb | XGBoost | 807 | 315 | 492 | 39.03% | 35.83% | 36.04% | 10.97 pp | -177 | 46 | -3.85 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 974 | 462 | 512 | 47.43% | 49.17% | 46.46% | 2.57 pp | -50 | 51 | -0.98 |
| BTC Hourly | transformer | Transformer | 974 | 453 | 521 | 46.51% | 44.58% | 43.75% | 3.49 pp | -68 | 51 | -1.33 |
| BTC Hourly | rf | RandomForest | 974 | 432 | 542 | 44.35% | 42.50% | 42.92% | 5.65 pp | -110 | 51 | -2.16 |
| BTC Hourly | nn | NN | 974 | 430 | 544 | 44.15% | 41.25% | 42.50% | 5.85 pp | -114 | 51 | -2.24 |
| BTC Hourly | lstm | LSTM | 974 | 414 | 560 | 42.51% | 37.92% | 41.04% | 7.49 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 974 | 402 | 572 | 41.27% | 35.83% | 38.96% | 8.73 pp | -170 | 51 | -3.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 797 | 385 | 412 | 48.31% | 47.08% | 47.92% | 1.69 pp | -27 | 46 | -0.59 |
| BTC Daily | nn | NN | 797 | 370 | 427 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 797 | 370 | 427 | 46.42% | 40.00% | 46.25% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | lstm | LSTM | 797 | 335 | 462 | 42.03% | 34.17% | 40.00% | 7.97 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 797 | 333 | 464 | 41.78% | 37.92% | 41.46% | 8.22 pp | -131 | 46 | -2.85 |
| BTC Daily | xgb | XGBoost | 807 | 315 | 492 | 39.03% | 35.83% | 36.04% | 10.97 pp | -177 | 46 | -3.85 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 569 | 277 | 292 | 48.68% | 47.50% | 48.12% | 1.32 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 569 | 270 | 299 | 47.45% | 51.67% | 49.17% | 2.55 pp | -29 | 53 | -0.55 |
| BTC Market Hours | transformer | Transformer | 569 | 267 | 302 | 46.92% | 46.67% | 46.88% | 3.08 pp | -35 | 53 | -0.66 |
| BTC Market Hours | lstm | LSTM | 569 | 247 | 322 | 43.41% | 42.50% | 43.96% | 6.59 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 569 | 245 | 324 | 43.06% | 45.42% | 43.75% | 6.94 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 569 | 244 | 325 | 42.88% | 46.25% | 43.33% | 7.12 pp | -81 | 53 | -1.53 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 622 | 291 | 331 | 46.78% | 49.17% | 47.71% | 3.22 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 622 | 290 | 332 | 46.62% | 47.50% | 47.92% | 3.38 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 622 | 289 | 333 | 46.46% | 48.75% | 47.08% | 3.54 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 622 | 260 | 362 | 41.80% | 43.75% | 40.83% | 8.20 pp | -102 | 53 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 622 | 257 | 365 | 41.32% | 44.17% | 40.83% | 8.68 pp | -108 | 53 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 622 | 253 | 369 | 40.68% | 40.83% | 39.79% | 9.32 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
