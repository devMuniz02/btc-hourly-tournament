# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T18:57:10.190495+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1297 | 1009 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1172 | 807 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 17:00:00+00:00 | 898 | 569 | 328 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 17:00:00+00:00 | 900 | 623 | 275 | 2 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 797 | 384 | 413 | 48.18% | 47.08% | 47.71% | 1.82 pp | -29 | 46 | -0.63 |
| BTC Market Hours | transformer | Transformer | 569 | 267 | 302 | 46.92% | 46.67% | 46.88% | 3.08 pp | -35 | 53 | -0.66 |
| BTC Market Hours Daily | nn | NN | 623 | 291 | 332 | 46.71% | 47.92% | 48.12% | 3.29 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 623 | 291 | 332 | 46.71% | 49.17% | 47.71% | 3.29 pp | -41 | 53 | -0.77 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 623 | 290 | 333 | 46.55% | 49.17% | 47.29% | 3.45 pp | -43 | 53 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 975 | 463 | 512 | 47.49% | 49.58% | 46.67% | 2.51 pp | -49 | 51 | -0.96 |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 797 | 370 | 427 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 797 | 369 | 428 | 46.30% | 39.58% | 46.04% | 3.70 pp | -59 | 46 | -1.28 |
| BTC Hourly | transformer | Transformer | 975 | 453 | 522 | 46.46% | 44.17% | 43.75% | 3.54 pp | -69 | 51 | -1.35 |
| BTC Market Hours | lstm | LSTM | 569 | 247 | 322 | 43.41% | 42.50% | 43.96% | 6.59 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 569 | 245 | 324 | 43.06% | 45.42% | 43.75% | 6.94 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 569 | 244 | 325 | 42.88% | 46.25% | 43.33% | 7.12 pp | -81 | 53 | -1.53 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 623 | 260 | 363 | 41.73% | 43.75% | 40.83% | 8.27 pp | -103 | 53 | -1.94 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 623 | 257 | 366 | 41.25% | 44.17% | 40.83% | 8.75 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 623 | 254 | 369 | 40.77% | 41.25% | 40.00% | 9.23 pp | -115 | 53 | -2.17 |
| BTC Hourly | rf | RandomForest | 975 | 432 | 543 | 44.31% | 42.08% | 42.92% | 5.69 pp | -111 | 51 | -2.18 |
| BTC Hourly | nn | NN | 975 | 431 | 544 | 44.21% | 41.25% | 42.50% | 5.79 pp | -113 | 51 | -2.22 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 797 | 336 | 461 | 42.16% | 34.17% | 40.21% | 7.84 pp | -125 | 46 | -2.72 |
| BTC Hourly | lstm | LSTM | 975 | 415 | 560 | 42.56% | 37.92% | 41.04% | 7.44 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 797 | 333 | 464 | 41.78% | 37.92% | 41.46% | 8.22 pp | -131 | 46 | -2.85 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 975 | 403 | 572 | 41.33% | 35.83% | 39.17% | 8.67 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 807 | 314 | 493 | 38.91% | 35.42% | 35.83% | 11.09 pp | -179 | 46 | -3.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 975 | 463 | 512 | 47.49% | 49.58% | 46.67% | 2.51 pp | -49 | 51 | -0.96 |
| BTC Hourly | transformer | Transformer | 975 | 453 | 522 | 46.46% | 44.17% | 43.75% | 3.54 pp | -69 | 51 | -1.35 |
| BTC Hourly | rf | RandomForest | 975 | 432 | 543 | 44.31% | 42.08% | 42.92% | 5.69 pp | -111 | 51 | -2.18 |
| BTC Hourly | nn | NN | 975 | 431 | 544 | 44.21% | 41.25% | 42.50% | 5.79 pp | -113 | 51 | -2.22 |
| BTC Hourly | lstm | LSTM | 975 | 415 | 560 | 42.56% | 37.92% | 41.04% | 7.44 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 975 | 403 | 572 | 41.33% | 35.83% | 39.17% | 8.67 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 797 | 384 | 413 | 48.18% | 47.08% | 47.71% | 1.82 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 797 | 370 | 427 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 797 | 369 | 428 | 46.30% | 39.58% | 46.04% | 3.70 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 797 | 336 | 461 | 42.16% | 34.17% | 40.21% | 7.84 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 797 | 333 | 464 | 41.78% | 37.92% | 41.46% | 8.22 pp | -131 | 46 | -2.85 |
| BTC Daily | xgb | XGBoost | 807 | 314 | 493 | 38.91% | 35.42% | 35.83% | 11.09 pp | -179 | 46 | -3.89 |

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
| BTC Market Hours Daily | nn | NN | 623 | 291 | 332 | 46.71% | 47.92% | 48.12% | 3.29 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 623 | 291 | 332 | 46.71% | 49.17% | 47.71% | 3.29 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 623 | 290 | 333 | 46.55% | 49.17% | 47.29% | 3.45 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 623 | 260 | 363 | 41.73% | 43.75% | 40.83% | 8.27 pp | -103 | 53 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 623 | 257 | 366 | 41.25% | 44.17% | 40.83% | 8.75 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 623 | 254 | 369 | 40.77% | 41.25% | 40.00% | 9.23 pp | -115 | 53 | -2.17 |

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
