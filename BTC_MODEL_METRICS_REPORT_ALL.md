# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T06:59:32.438363+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1191 | 903 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1067 | 702 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 709 | 464 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 711 | 518 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T14:00:00+00:00 | 114 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T14:00:00+00:00 | 114 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T14:00:00+00:00 | 114 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T14:00:00+00:00 | 115 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 114 | 57 | 57 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 114 | 57 | 57 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 464 | 225 | 239 | 48.49% | 44.17% | 48.49% | 1.51 pp | -14 | 45 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 692 | 338 | 354 | 48.84% | 45.42% | 49.38% | 1.16 pp | -16 | 42 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 464 | 218 | 246 | 46.98% | 48.33% | 46.98% | 3.02 pp | -28 | 45 | -0.62 |
| BTC Daily | transformer | Transformer | 692 | 332 | 360 | 47.98% | 46.25% | 49.17% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Market Hours | transformer | Transformer | 464 | 215 | 249 | 46.34% | 40.00% | 46.34% | 3.66 pp | -34 | 45 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 518 | 238 | 280 | 45.95% | 46.25% | 46.25% | 4.05 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | nn | NN | 518 | 237 | 281 | 45.75% | 43.33% | 46.46% | 4.25 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 518 | 237 | 281 | 45.75% | 47.92% | 46.25% | 4.25 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 869 | 411 | 458 | 47.30% | 47.08% | 47.71% | 2.70 pp | -47 | 46 | -1.02 |
| BTC Daily | nn | NN | 692 | 322 | 370 | 46.53% | 42.92% | 48.75% | 3.47 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 869 | 408 | 461 | 46.95% | 47.08% | 47.08% | 3.05 pp | -53 | 46 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 10 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 10 | -1.40 |
| BTC Market Hours | rf | RandomForest | 464 | 200 | 264 | 43.10% | 43.33% | 43.10% | 6.90 pp | -64 | 45 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 464 | 197 | 267 | 42.46% | 40.42% | 42.46% | 7.54 pp | -70 | 45 | -1.56 |
| BTC Hourly | nn | NN | 869 | 391 | 478 | 44.99% | 45.83% | 44.17% | 5.01 pp | -87 | 46 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 45 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 518 | 215 | 303 | 41.51% | 42.08% | 41.67% | 8.49 pp | -88 | 45 | -1.96 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 869 | 388 | 481 | 44.65% | 45.00% | 44.38% | 5.35 pp | -93 | 46 | -2.02 |
| BTC Daily | lstm | LSTM | 692 | 301 | 391 | 43.50% | 38.75% | 42.50% | 6.50 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 518 | 207 | 311 | 39.96% | 38.33% | 40.62% | 10.04 pp | -104 | 45 | -2.31 |
| BTC Daily | rf | RandomForest | 692 | 297 | 395 | 42.92% | 40.42% | 43.33% | 7.08 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 518 | 206 | 312 | 39.77% | 37.50% | 39.58% | 10.23 pp | -106 | 45 | -2.36 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 869 | 370 | 499 | 42.58% | 38.75% | 42.08% | 7.42 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 869 | 367 | 502 | 42.23% | 41.25% | 43.33% | 7.77 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 702 | 278 | 424 | 39.60% | 35.83% | 39.38% | 10.40 pp | -146 | 42 | -3.48 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 869 | 411 | 458 | 47.30% | 47.08% | 47.71% | 2.70 pp | -47 | 46 | -1.02 |
| BTC Hourly | transformer | Transformer | 869 | 408 | 461 | 46.95% | 47.08% | 47.08% | 3.05 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 869 | 391 | 478 | 44.99% | 45.83% | 44.17% | 5.01 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 869 | 388 | 481 | 44.65% | 45.00% | 44.38% | 5.35 pp | -93 | 46 | -2.02 |
| BTC Hourly | lstm | LSTM | 869 | 370 | 499 | 42.58% | 38.75% | 42.08% | 7.42 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 869 | 367 | 502 | 42.23% | 41.25% | 43.33% | 7.77 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 692 | 338 | 354 | 48.84% | 45.42% | 49.38% | 1.16 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 692 | 332 | 360 | 47.98% | 46.25% | 49.17% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 692 | 322 | 370 | 46.53% | 42.92% | 48.75% | 3.47 pp | -48 | 42 | -1.14 |
| BTC Daily | lstm | LSTM | 692 | 301 | 391 | 43.50% | 38.75% | 42.50% | 6.50 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 692 | 297 | 395 | 42.92% | 40.42% | 43.33% | 7.08 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 702 | 278 | 424 | 39.60% | 35.83% | 39.38% | 10.40 pp | -146 | 42 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 464 | 225 | 239 | 48.49% | 44.17% | 48.49% | 1.51 pp | -14 | 45 | -0.31 |
| BTC Market Hours | nn | NN | 464 | 218 | 246 | 46.98% | 48.33% | 46.98% | 3.02 pp | -28 | 45 | -0.62 |
| BTC Market Hours | transformer | Transformer | 464 | 215 | 249 | 46.34% | 40.00% | 46.34% | 3.66 pp | -34 | 45 | -0.76 |
| BTC Market Hours | rf | RandomForest | 464 | 200 | 264 | 43.10% | 43.33% | 43.10% | 6.90 pp | -64 | 45 | -1.42 |
| BTC Market Hours | lstm | LSTM | 464 | 197 | 267 | 42.46% | 40.42% | 42.46% | 7.54 pp | -70 | 45 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 45 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 518 | 238 | 280 | 45.95% | 46.25% | 46.25% | 4.05 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | nn | NN | 518 | 237 | 281 | 45.75% | 43.33% | 46.46% | 4.25 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 518 | 237 | 281 | 45.75% | 47.92% | 46.25% | 4.25 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 518 | 215 | 303 | 41.51% | 42.08% | 41.67% | 8.49 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 518 | 207 | 311 | 39.96% | 38.33% | 40.62% | 10.04 pp | -104 | 45 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 518 | 206 | 312 | 39.77% | 37.50% | 39.58% | 10.23 pp | -106 | 45 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 114 | 57 | 57 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 10 | -1.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 114 | 57 | 57 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 114 | 55 | 59 | 48.25% | 48.25% | 48.25% | 1.75 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 114 | 52 | 62 | 45.61% | 45.61% | 45.61% | 4.39 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 10 | -1.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
