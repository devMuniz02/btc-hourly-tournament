# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T12:17:58.960102+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1292 | 1004 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1168 | 803 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 888 | 565 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 890 | 619 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 208 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 565 | 274 | 291 | 48.50% | 47.08% | 47.71% | 1.50 pp | -17 | 53 | -0.32 |
| Consolidated Hourly | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
| BTC Market Hours | nn | NN | 565 | 267 | 298 | 47.26% | 50.42% | 48.75% | 2.74 pp | -31 | 53 | -0.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 793 | 383 | 410 | 48.30% | 46.67% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 565 | 266 | 299 | 47.08% | 46.67% | 47.08% | 2.92 pp | -33 | 53 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 619 | 290 | 329 | 46.85% | 49.58% | 47.71% | 3.15 pp | -39 | 53 | -0.74 |
| BTC Market Hours Daily | nn | NN | 619 | 288 | 331 | 46.53% | 47.08% | 47.71% | 3.47 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 619 | 287 | 332 | 46.37% | 48.33% | 46.88% | 3.63 pp | -45 | 53 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 970 | 460 | 510 | 47.42% | 48.75% | 46.04% | 2.58 pp | -50 | 50 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 793 | 369 | 424 | 46.53% | 45.42% | 45.00% | 3.47 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 793 | 369 | 424 | 46.53% | 40.42% | 46.46% | 3.47 pp | -55 | 46 | -1.20 |
| BTC Hourly | transformer | Transformer | 970 | 452 | 518 | 46.60% | 44.17% | 43.75% | 3.40 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| BTC Market Hours | lstm | LSTM | 565 | 245 | 320 | 43.36% | 42.08% | 43.96% | 6.64 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 565 | 243 | 322 | 43.01% | 45.42% | 43.54% | 6.99 pp | -79 | 53 | -1.49 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 565 | 242 | 323 | 42.83% | 46.25% | 43.54% | 7.17 pp | -81 | 53 | -1.53 |
| Consolidated Hourly | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 619 | 258 | 361 | 41.68% | 43.75% | 40.62% | 8.32 pp | -103 | 53 | -1.94 |
| Consolidated Hourly | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 619 | 254 | 365 | 41.03% | 43.75% | 40.62% | 8.97 pp | -111 | 53 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 619 | 252 | 367 | 40.71% | 40.83% | 40.00% | 9.29 pp | -115 | 53 | -2.17 |
| BTC Hourly | rf | RandomForest | 970 | 429 | 541 | 44.23% | 41.67% | 42.50% | 5.77 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 970 | 428 | 542 | 44.12% | 41.25% | 42.50% | 5.88 pp | -114 | 50 | -2.28 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 793 | 333 | 460 | 41.99% | 33.75% | 39.79% | 8.01 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Daily | rf | RandomForest | 793 | 331 | 462 | 41.74% | 37.92% | 41.46% | 8.26 pp | -131 | 46 | -2.85 |
| BTC Hourly | lstm | LSTM | 970 | 412 | 558 | 42.47% | 37.50% | 41.04% | 7.53 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 970 | 400 | 570 | 41.24% | 36.25% | 38.54% | 8.76 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 803 | 314 | 489 | 39.10% | 35.83% | 36.04% | 10.90 pp | -175 | 46 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 970 | 460 | 510 | 47.42% | 48.75% | 46.04% | 2.58 pp | -50 | 50 | -1.00 |
| BTC Hourly | transformer | Transformer | 970 | 452 | 518 | 46.60% | 44.17% | 43.75% | 3.40 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 970 | 429 | 541 | 44.23% | 41.67% | 42.50% | 5.77 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 970 | 428 | 542 | 44.12% | 41.25% | 42.50% | 5.88 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 970 | 412 | 558 | 42.47% | 37.50% | 41.04% | 7.53 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 970 | 400 | 570 | 41.24% | 36.25% | 38.54% | 8.76 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 793 | 383 | 410 | 48.30% | 46.67% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Daily | nn | NN | 793 | 369 | 424 | 46.53% | 45.42% | 45.00% | 3.47 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 793 | 369 | 424 | 46.53% | 40.42% | 46.46% | 3.47 pp | -55 | 46 | -1.20 |
| BTC Daily | lstm | LSTM | 793 | 333 | 460 | 41.99% | 33.75% | 39.79% | 8.01 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 793 | 331 | 462 | 41.74% | 37.92% | 41.46% | 8.26 pp | -131 | 46 | -2.85 |
| BTC Daily | xgb | XGBoost | 803 | 314 | 489 | 39.10% | 35.83% | 36.04% | 10.90 pp | -175 | 46 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 565 | 274 | 291 | 48.50% | 47.08% | 47.71% | 1.50 pp | -17 | 53 | -0.32 |
| BTC Market Hours | nn | NN | 565 | 267 | 298 | 47.26% | 50.42% | 48.75% | 2.74 pp | -31 | 53 | -0.58 |
| BTC Market Hours | transformer | Transformer | 565 | 266 | 299 | 47.08% | 46.67% | 47.08% | 2.92 pp | -33 | 53 | -0.62 |
| BTC Market Hours | lstm | LSTM | 565 | 245 | 320 | 43.36% | 42.08% | 43.96% | 6.64 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 565 | 243 | 322 | 43.01% | 45.42% | 43.54% | 6.99 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 565 | 242 | 323 | 42.83% | 46.25% | 43.54% | 7.17 pp | -81 | 53 | -1.53 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 619 | 290 | 329 | 46.85% | 49.58% | 47.71% | 3.15 pp | -39 | 53 | -0.74 |
| BTC Market Hours Daily | nn | NN | 619 | 288 | 331 | 46.53% | 47.08% | 47.71% | 3.47 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 619 | 287 | 332 | 46.37% | 48.33% | 46.88% | 3.63 pp | -45 | 53 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 619 | 258 | 361 | 41.68% | 43.75% | 40.62% | 8.32 pp | -103 | 53 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 619 | 254 | 365 | 41.03% | 43.75% | 40.62% | 8.97 pp | -111 | 53 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 619 | 252 | 367 | 40.71% | 40.83% | 40.00% | 9.29 pp | -115 | 53 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
