# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T18:21:50.372039+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1199 | 911 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1074 | 709 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 722 | 471 | 250 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 17:00:00+00:00 | 724 | 525 | 197 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 471 | 228 | 243 | 48.41% | 44.17% | 48.41% | 1.59 pp | -15 | 46 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 699 | 342 | 357 | 48.93% | 47.08% | 49.17% | 1.07 pp | -15 | 42 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 471 | 222 | 249 | 47.13% | 47.92% | 47.13% | 2.87 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 699 | 336 | 363 | 48.07% | 47.08% | 49.58% | 1.93 pp | -27 | 42 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 471 | 219 | 252 | 46.50% | 41.25% | 46.50% | 3.50 pp | -33 | 46 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 525 | 241 | 284 | 45.90% | 47.08% | 46.67% | 4.10 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 525 | 241 | 284 | 45.90% | 47.50% | 46.46% | 4.10 pp | -43 | 46 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 877 | 416 | 461 | 47.43% | 48.75% | 47.92% | 2.57 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 525 | 240 | 285 | 45.71% | 42.92% | 46.46% | 4.29 pp | -45 | 46 | -0.98 |
| BTC Hourly | transformer | Transformer | 877 | 414 | 463 | 47.21% | 47.92% | 47.71% | 2.79 pp | -49 | 47 | -1.04 |
| BTC Daily | nn | NN | 699 | 325 | 374 | 46.49% | 43.33% | 48.75% | 3.51 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 471 | 203 | 268 | 43.10% | 42.92% | 43.10% | 6.90 pp | -65 | 46 | -1.41 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 471 | 201 | 270 | 42.68% | 40.42% | 42.68% | 7.32 pp | -69 | 46 | -1.50 |
| BTC Hourly | nn | NN | 877 | 395 | 482 | 45.04% | 46.67% | 43.96% | 4.96 pp | -87 | 47 | -1.85 |
| BTC Market Hours | xgb | XGBoost | 471 | 192 | 279 | 40.76% | 40.42% | 40.76% | 9.24 pp | -87 | 46 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 525 | 217 | 308 | 41.33% | 41.25% | 41.46% | 8.67 pp | -91 | 46 | -1.98 |
| BTC Hourly | rf | RandomForest | 877 | 391 | 486 | 44.58% | 45.00% | 44.38% | 5.42 pp | -95 | 47 | -2.02 |
| BTC Daily | lstm | LSTM | 699 | 304 | 395 | 43.49% | 38.75% | 42.29% | 6.51 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 525 | 210 | 315 | 40.00% | 37.08% | 41.04% | 10.00 pp | -105 | 46 | -2.28 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Daily | rf | RandomForest | 699 | 300 | 399 | 42.92% | 41.25% | 43.33% | 7.08 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 525 | 208 | 317 | 39.62% | 37.92% | 39.17% | 10.38 pp | -109 | 46 | -2.37 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 877 | 374 | 503 | 42.65% | 38.33% | 41.88% | 7.35 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 877 | 372 | 505 | 42.42% | 42.08% | 43.33% | 7.58 pp | -133 | 47 | -2.83 |
| BTC Daily | xgb | XGBoost | 709 | 281 | 428 | 39.63% | 35.83% | 39.38% | 10.37 pp | -147 | 42 | -3.50 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 877 | 416 | 461 | 47.43% | 48.75% | 47.92% | 2.57 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 877 | 414 | 463 | 47.21% | 47.92% | 47.71% | 2.79 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 877 | 395 | 482 | 45.04% | 46.67% | 43.96% | 4.96 pp | -87 | 47 | -1.85 |
| BTC Hourly | rf | RandomForest | 877 | 391 | 486 | 44.58% | 45.00% | 44.38% | 5.42 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 877 | 374 | 503 | 42.65% | 38.33% | 41.88% | 7.35 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 877 | 372 | 505 | 42.42% | 42.08% | 43.33% | 7.58 pp | -133 | 47 | -2.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 699 | 342 | 357 | 48.93% | 47.08% | 49.17% | 1.07 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 699 | 336 | 363 | 48.07% | 47.08% | 49.58% | 1.93 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 699 | 325 | 374 | 46.49% | 43.33% | 48.75% | 3.51 pp | -49 | 42 | -1.17 |
| BTC Daily | lstm | LSTM | 699 | 304 | 395 | 43.49% | 38.75% | 42.29% | 6.51 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 699 | 300 | 399 | 42.92% | 41.25% | 43.33% | 7.08 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 709 | 281 | 428 | 39.63% | 35.83% | 39.38% | 10.37 pp | -147 | 42 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 471 | 228 | 243 | 48.41% | 44.17% | 48.41% | 1.59 pp | -15 | 46 | -0.33 |
| BTC Market Hours | nn | NN | 471 | 222 | 249 | 47.13% | 47.92% | 47.13% | 2.87 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 471 | 219 | 252 | 46.50% | 41.25% | 46.50% | 3.50 pp | -33 | 46 | -0.72 |
| BTC Market Hours | rf | RandomForest | 471 | 203 | 268 | 43.10% | 42.92% | 43.10% | 6.90 pp | -65 | 46 | -1.41 |
| BTC Market Hours | lstm | LSTM | 471 | 201 | 270 | 42.68% | 40.42% | 42.68% | 7.32 pp | -69 | 46 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 471 | 192 | 279 | 40.76% | 40.42% | 40.76% | 9.24 pp | -87 | 46 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 525 | 241 | 284 | 45.90% | 47.08% | 46.67% | 4.10 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 525 | 241 | 284 | 45.90% | 47.50% | 46.46% | 4.10 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | nn | NN | 525 | 240 | 285 | 45.71% | 42.92% | 46.46% | 4.29 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 525 | 217 | 308 | 41.33% | 41.25% | 41.46% | 8.67 pp | -91 | 46 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 525 | 210 | 315 | 40.00% | 37.08% | 41.04% | 10.00 pp | -105 | 46 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 525 | 208 | 317 | 39.62% | 37.92% | 39.17% | 10.38 pp | -109 | 46 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
