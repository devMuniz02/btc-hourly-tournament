# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T13:25:20.843466+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1163 | 875 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1039 | 674 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 12:00:00+00:00 | 656 | 436 | 219 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 12:00:00+00:00 | 657 | 489 | 166 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 436 | 215 | 221 | 49.31% | 45.42% | 49.31% | 0.69 pp | -6 | 43 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 326 | 338 | 49.10% | 47.50% | 50.21% | 0.90 pp | -12 | 40 | -0.30 |
| BTC Daily | transformer | Transformer | 664 | 320 | 344 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Market Hours | nn | NN | 436 | 205 | 231 | 47.02% | 49.17% | 47.02% | 2.98 pp | -26 | 43 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 436 | 200 | 236 | 45.87% | 40.83% | 45.87% | 4.13 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 489 | 226 | 263 | 46.22% | 47.08% | 46.46% | 3.78 pp | -37 | 43 | -0.86 |
| BTC Daily | nn | NN | 664 | 313 | 351 | 47.14% | 43.75% | 49.79% | 2.86 pp | -38 | 40 | -0.95 |
| BTC Hourly | transformer | Transformer | 841 | 399 | 442 | 47.44% | 47.92% | 47.08% | 2.56 pp | -43 | 45 | -0.96 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 489 | 222 | 267 | 45.40% | 42.92% | 45.83% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 489 | 222 | 267 | 45.40% | 45.00% | 45.62% | 4.60 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 841 | 394 | 447 | 46.85% | 43.33% | 46.46% | 3.15 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 436 | 188 | 248 | 43.12% | 42.50% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 436 | 188 | 248 | 43.12% | 43.33% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| Consolidated Hourly | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 841 | 380 | 461 | 45.18% | 43.75% | 44.58% | 4.82 pp | -81 | 45 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 489 | 202 | 287 | 41.31% | 41.67% | 41.46% | 8.69 pp | -85 | 43 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 436 | 175 | 261 | 40.14% | 38.33% | 40.14% | 9.86 pp | -86 | 43 | -2.00 |
| BTC Hourly | rf | RandomForest | 841 | 375 | 466 | 44.59% | 43.33% | 43.96% | 5.41 pp | -91 | 45 | -2.02 |
| BTC Daily | lstm | LSTM | 664 | 291 | 373 | 43.83% | 39.17% | 43.12% | 6.17 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 489 | 196 | 293 | 40.08% | 38.75% | 40.42% | 9.92 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 664 | 285 | 379 | 42.92% | 41.25% | 44.17% | 7.08 pp | -94 | 40 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 489 | 191 | 298 | 39.06% | 35.83% | 39.38% | 10.94 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 841 | 361 | 480 | 42.93% | 39.58% | 42.29% | 7.07 pp | -119 | 45 | -2.64 |
| BTC Hourly | xgb | XGBoost | 841 | 355 | 486 | 42.21% | 39.58% | 42.50% | 7.79 pp | -131 | 45 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 674 | 269 | 405 | 39.91% | 34.58% | 40.21% | 10.09 pp | -136 | 40 | -3.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 841 | 399 | 442 | 47.44% | 47.92% | 47.08% | 2.56 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 841 | 394 | 447 | 46.85% | 43.33% | 46.46% | 3.15 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 841 | 380 | 461 | 45.18% | 43.75% | 44.58% | 4.82 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 841 | 375 | 466 | 44.59% | 43.33% | 43.96% | 5.41 pp | -91 | 45 | -2.02 |
| BTC Hourly | lstm | LSTM | 841 | 361 | 480 | 42.93% | 39.58% | 42.29% | 7.07 pp | -119 | 45 | -2.64 |
| BTC Hourly | xgb | XGBoost | 841 | 355 | 486 | 42.21% | 39.58% | 42.50% | 7.79 pp | -131 | 45 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 326 | 338 | 49.10% | 47.50% | 50.21% | 0.90 pp | -12 | 40 | -0.30 |
| BTC Daily | transformer | Transformer | 664 | 320 | 344 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Daily | nn | NN | 664 | 313 | 351 | 47.14% | 43.75% | 49.79% | 2.86 pp | -38 | 40 | -0.95 |
| BTC Daily | lstm | LSTM | 664 | 291 | 373 | 43.83% | 39.17% | 43.12% | 6.17 pp | -82 | 40 | -2.05 |
| BTC Daily | rf | RandomForest | 664 | 285 | 379 | 42.92% | 41.25% | 44.17% | 7.08 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 674 | 269 | 405 | 39.91% | 34.58% | 40.21% | 10.09 pp | -136 | 40 | -3.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 436 | 215 | 221 | 49.31% | 45.42% | 49.31% | 0.69 pp | -6 | 43 | -0.14 |
| BTC Market Hours | nn | NN | 436 | 205 | 231 | 47.02% | 49.17% | 47.02% | 2.98 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 436 | 200 | 236 | 45.87% | 40.83% | 45.87% | 4.13 pp | -36 | 43 | -0.84 |
| BTC Market Hours | lstm | LSTM | 436 | 188 | 248 | 43.12% | 42.50% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 436 | 188 | 248 | 43.12% | 43.33% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 436 | 175 | 261 | 40.14% | 38.33% | 40.14% | 9.86 pp | -86 | 43 | -2.00 |

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
