# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T13:18:10.532956+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1293 | 1005 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1169 | 804 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 12:00:00+00:00 | 890 | 566 | 323 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 12:00:00+00:00 | 891 | 619 | 270 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 566 | 275 | 291 | 48.59% | 47.08% | 47.71% | 1.41 pp | -16 | 53 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 384 | 410 | 48.36% | 46.67% | 47.92% | 1.64 pp | -26 | 46 | -0.57 |
| BTC Market Hours | nn | NN | 566 | 268 | 298 | 47.35% | 50.83% | 48.75% | 2.65 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 566 | 266 | 300 | 47.00% | 46.67% | 46.88% | 3.00 pp | -34 | 53 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 619 | 290 | 329 | 46.85% | 49.58% | 47.71% | 3.15 pp | -39 | 53 | -0.74 |
| BTC Market Hours Daily | nn | NN | 619 | 288 | 331 | 46.53% | 47.08% | 47.71% | 3.47 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 619 | 287 | 332 | 46.37% | 48.33% | 46.88% | 3.63 pp | -45 | 53 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 971 | 460 | 511 | 47.37% | 48.75% | 46.04% | 2.63 pp | -51 | 50 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 794 | 369 | 425 | 46.47% | 45.00% | 45.00% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.00% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 971 | 452 | 519 | 46.55% | 44.17% | 43.75% | 3.45 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 566 | 245 | 321 | 43.29% | 42.08% | 43.75% | 6.71 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 566 | 244 | 322 | 43.11% | 45.42% | 43.54% | 6.89 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 566 | 243 | 323 | 42.93% | 46.67% | 43.54% | 7.07 pp | -80 | 53 | -1.51 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 619 | 258 | 361 | 41.68% | 43.75% | 40.62% | 8.32 pp | -103 | 53 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 619 | 254 | 365 | 41.03% | 43.75% | 40.62% | 8.97 pp | -111 | 53 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 619 | 252 | 367 | 40.71% | 40.83% | 40.00% | 9.29 pp | -115 | 53 | -2.17 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| BTC Hourly | nn | NN | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 794 | 333 | 461 | 41.94% | 33.75% | 39.79% | 8.06 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.50% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Hourly | lstm | LSTM | 971 | 413 | 558 | 42.53% | 37.50% | 41.04% | 7.47 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 971 | 400 | 571 | 41.19% | 35.83% | 38.54% | 8.81 pp | -171 | 50 | -3.42 |
| BTC Daily | xgb | XGBoost | 804 | 314 | 490 | 39.05% | 35.42% | 36.04% | 10.95 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 971 | 460 | 511 | 47.37% | 48.75% | 46.04% | 2.63 pp | -51 | 50 | -1.02 |
| BTC Hourly | transformer | Transformer | 971 | 452 | 519 | 46.55% | 44.17% | 43.75% | 3.45 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | lstm | LSTM | 971 | 413 | 558 | 42.53% | 37.50% | 41.04% | 7.47 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 971 | 400 | 571 | 41.19% | 35.83% | 38.54% | 8.81 pp | -171 | 50 | -3.42 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 384 | 410 | 48.36% | 46.67% | 47.92% | 1.64 pp | -26 | 46 | -0.57 |
| BTC Daily | nn | NN | 794 | 369 | 425 | 46.47% | 45.00% | 45.00% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.00% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 794 | 333 | 461 | 41.94% | 33.75% | 39.79% | 8.06 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.50% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 804 | 314 | 490 | 39.05% | 35.42% | 36.04% | 10.95 pp | -176 | 46 | -3.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 566 | 275 | 291 | 48.59% | 47.08% | 47.71% | 1.41 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 566 | 268 | 298 | 47.35% | 50.83% | 48.75% | 2.65 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 566 | 266 | 300 | 47.00% | 46.67% | 46.88% | 3.00 pp | -34 | 53 | -0.64 |
| BTC Market Hours | lstm | LSTM | 566 | 245 | 321 | 43.29% | 42.08% | 43.75% | 6.71 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 566 | 244 | 322 | 43.11% | 45.42% | 43.54% | 6.89 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 566 | 243 | 323 | 42.93% | 46.67% | 43.54% | 7.07 pp | -80 | 53 | -1.51 |

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
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
