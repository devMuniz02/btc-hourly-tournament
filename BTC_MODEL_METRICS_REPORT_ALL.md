# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T12:11:37.922901+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1162 | 874 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1038 | 673 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 654 | 435 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 656 | 489 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 87 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 87 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 3 | 84 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 3 | 84 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 435 | 214 | 221 | 49.20% | 45.42% | 49.20% | 0.80 pp | -7 | 43 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 663 | 325 | 338 | 49.02% | 47.08% | 50.21% | 0.98 pp | -13 | 40 | -0.33 |
| BTC Daily | transformer | Transformer | 663 | 320 | 343 | 48.27% | 45.00% | 49.58% | 1.73 pp | -23 | 40 | -0.57 |
| BTC Market Hours | nn | NN | 435 | 204 | 231 | 46.90% | 48.75% | 46.90% | 3.10 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 489 | 226 | 263 | 46.22% | 47.08% | 46.46% | 3.78 pp | -37 | 43 | -0.86 |
| BTC Market Hours | transformer | Transformer | 435 | 199 | 236 | 45.75% | 40.83% | 45.75% | 4.25 pp | -37 | 43 | -0.86 |
| BTC Hourly | transformer | Transformer | 840 | 399 | 441 | 47.50% | 48.33% | 47.08% | 2.50 pp | -42 | 45 | -0.93 |
| BTC Daily | nn | NN | 663 | 312 | 351 | 47.06% | 43.33% | 49.79% | 2.94 pp | -39 | 40 | -0.97 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 489 | 222 | 267 | 45.40% | 42.92% | 45.83% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 489 | 222 | 267 | 45.40% | 45.00% | 45.62% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 840 | 393 | 447 | 46.79% | 43.33% | 46.46% | 3.21 pp | -54 | 45 | -1.20 |
| BTC Market Hours | lstm | LSTM | 435 | 187 | 248 | 42.99% | 42.50% | 42.99% | 7.01 pp | -61 | 43 | -1.42 |
| BTC Market Hours | rf | RandomForest | 435 | 187 | 248 | 42.99% | 43.33% | 42.99% | 7.01 pp | -61 | 43 | -1.42 |
| Consolidated Hourly | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 840 | 380 | 460 | 45.24% | 44.17% | 44.79% | 4.76 pp | -80 | 45 | -1.78 |
| BTC Market Hours Daily | rf | RandomForest | 489 | 202 | 287 | 41.31% | 41.67% | 41.46% | 8.69 pp | -85 | 43 | -1.98 |
| BTC Hourly | rf | RandomForest | 840 | 375 | 465 | 44.64% | 43.75% | 44.17% | 5.36 pp | -90 | 45 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 435 | 174 | 261 | 40.00% | 38.33% | 40.00% | 10.00 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 663 | 291 | 372 | 43.89% | 39.58% | 43.33% | 6.11 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 489 | 196 | 293 | 40.08% | 38.75% | 40.42% | 9.92 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 663 | 285 | 378 | 42.99% | 41.67% | 44.38% | 7.01 pp | -93 | 40 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 489 | 191 | 298 | 39.06% | 35.83% | 39.38% | 10.94 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 840 | 361 | 479 | 42.98% | 40.00% | 42.50% | 7.02 pp | -118 | 45 | -2.62 |
| BTC Hourly | xgb | XGBoost | 840 | 355 | 485 | 42.26% | 40.00% | 42.71% | 7.74 pp | -130 | 45 | -2.89 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 673 | 268 | 405 | 39.82% | 34.17% | 40.21% | 10.18 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 840 | 399 | 441 | 47.50% | 48.33% | 47.08% | 2.50 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 840 | 393 | 447 | 46.79% | 43.33% | 46.46% | 3.21 pp | -54 | 45 | -1.20 |
| BTC Hourly | nn | NN | 840 | 380 | 460 | 45.24% | 44.17% | 44.79% | 4.76 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 840 | 375 | 465 | 44.64% | 43.75% | 44.17% | 5.36 pp | -90 | 45 | -2.00 |
| BTC Hourly | lstm | LSTM | 840 | 361 | 479 | 42.98% | 40.00% | 42.50% | 7.02 pp | -118 | 45 | -2.62 |
| BTC Hourly | xgb | XGBoost | 840 | 355 | 485 | 42.26% | 40.00% | 42.71% | 7.74 pp | -130 | 45 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 663 | 325 | 338 | 49.02% | 47.08% | 50.21% | 0.98 pp | -13 | 40 | -0.33 |
| BTC Daily | transformer | Transformer | 663 | 320 | 343 | 48.27% | 45.00% | 49.58% | 1.73 pp | -23 | 40 | -0.57 |
| BTC Daily | nn | NN | 663 | 312 | 351 | 47.06% | 43.33% | 49.79% | 2.94 pp | -39 | 40 | -0.97 |
| BTC Daily | lstm | LSTM | 663 | 291 | 372 | 43.89% | 39.58% | 43.33% | 6.11 pp | -81 | 40 | -2.02 |
| BTC Daily | rf | RandomForest | 663 | 285 | 378 | 42.99% | 41.67% | 44.38% | 7.01 pp | -93 | 40 | -2.33 |
| BTC Daily | xgb | XGBoost | 673 | 268 | 405 | 39.82% | 34.17% | 40.21% | 10.18 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 435 | 214 | 221 | 49.20% | 45.42% | 49.20% | 0.80 pp | -7 | 43 | -0.16 |
| BTC Market Hours | nn | NN | 435 | 204 | 231 | 46.90% | 48.75% | 46.90% | 3.10 pp | -27 | 43 | -0.63 |
| BTC Market Hours | transformer | Transformer | 435 | 199 | 236 | 45.75% | 40.83% | 45.75% | 4.25 pp | -37 | 43 | -0.86 |
| BTC Market Hours | lstm | LSTM | 435 | 187 | 248 | 42.99% | 42.50% | 42.99% | 7.01 pp | -61 | 43 | -1.42 |
| BTC Market Hours | rf | RandomForest | 435 | 187 | 248 | 42.99% | 43.33% | 42.99% | 7.01 pp | -61 | 43 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 435 | 174 | 261 | 40.00% | 38.33% | 40.00% | 10.00 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 489 | 226 | 263 | 46.22% | 47.08% | 46.46% | 3.78 pp | -37 | 43 | -0.86 |
| BTC Market Hours Daily | nn | NN | 489 | 222 | 267 | 45.40% | 42.92% | 45.83% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 489 | 222 | 267 | 45.40% | 45.00% | 45.62% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 489 | 202 | 287 | 41.31% | 41.67% | 41.46% | 8.69 pp | -85 | 43 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 489 | 196 | 293 | 40.08% | 38.75% | 40.42% | 9.92 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 489 | 191 | 298 | 39.06% | 35.83% | 39.38% | 10.94 pp | -107 | 43 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
