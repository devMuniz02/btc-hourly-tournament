# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T07:51:18.113013+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1305 | 1017 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1181 | 816 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 914 | 578 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 916 | 632 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 578 | 280 | 298 | 48.44% | 46.67% | 47.50% | 1.56 pp | -18 | 54 | -0.33 |
| BTC Market Hours | nn | NN | 578 | 277 | 301 | 47.92% | 52.08% | 49.58% | 2.08 pp | -24 | 54 | -0.44 |
| BTC Market Hours | transformer | Transformer | 578 | 274 | 304 | 47.40% | 47.92% | 47.29% | 2.60 pp | -30 | 54 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 806 | 389 | 417 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Market Hours Daily | transformer | Transformer | 632 | 297 | 335 | 46.99% | 49.58% | 47.50% | 3.01 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 632 | 296 | 336 | 46.84% | 47.50% | 47.92% | 3.16 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 632 | 295 | 337 | 46.68% | 48.75% | 46.88% | 3.32 pp | -42 | 54 | -0.78 |
| Consolidated Hourly | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 983 | 469 | 514 | 47.71% | 50.83% | 46.88% | 2.29 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 806 | 375 | 431 | 46.53% | 44.58% | 45.21% | 3.47 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 806 | 374 | 432 | 46.40% | 39.58% | 46.25% | 3.60 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 983 | 455 | 528 | 46.29% | 43.75% | 43.54% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 578 | 249 | 329 | 43.08% | 42.08% | 43.12% | 6.92 pp | -80 | 54 | -1.48 |
| Consolidated Hourly | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 578 | 248 | 330 | 42.91% | 44.58% | 43.12% | 7.09 pp | -82 | 54 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 578 | 248 | 330 | 42.91% | 45.83% | 43.12% | 7.09 pp | -82 | 54 | -1.52 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 632 | 263 | 369 | 41.61% | 43.33% | 40.42% | 8.39 pp | -106 | 54 | -1.96 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 632 | 261 | 371 | 41.30% | 45.00% | 40.42% | 8.70 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 632 | 258 | 374 | 40.82% | 41.67% | 40.00% | 9.18 pp | -116 | 54 | -2.15 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 983 | 434 | 549 | 44.15% | 41.67% | 42.71% | 5.85 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 983 | 433 | 550 | 44.05% | 41.67% | 42.08% | 5.95 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |
| BTC Daily | lstm | LSTM | 806 | 340 | 466 | 42.18% | 35.42% | 40.42% | 7.82 pp | -126 | 46 | -2.74 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 983 | 418 | 565 | 42.52% | 37.92% | 40.62% | 7.48 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 806 | 335 | 471 | 41.56% | 37.08% | 41.25% | 8.44 pp | -136 | 46 | -2.96 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 983 | 405 | 578 | 41.20% | 35.83% | 38.96% | 8.80 pp | -173 | 51 | -3.39 |
| BTC Daily | xgb | XGBoost | 816 | 318 | 498 | 38.97% | 35.42% | 35.62% | 11.03 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 983 | 469 | 514 | 47.71% | 50.83% | 46.88% | 2.29 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 983 | 455 | 528 | 46.29% | 43.75% | 43.54% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 983 | 434 | 549 | 44.15% | 41.67% | 42.71% | 5.85 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 983 | 433 | 550 | 44.05% | 41.67% | 42.08% | 5.95 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 983 | 418 | 565 | 42.52% | 37.92% | 40.62% | 7.48 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 983 | 405 | 578 | 41.20% | 35.83% | 38.96% | 8.80 pp | -173 | 51 | -3.39 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 806 | 389 | 417 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 806 | 375 | 431 | 46.53% | 44.58% | 45.21% | 3.47 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 806 | 374 | 432 | 46.40% | 39.58% | 46.25% | 3.60 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 806 | 340 | 466 | 42.18% | 35.42% | 40.42% | 7.82 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 806 | 335 | 471 | 41.56% | 37.08% | 41.25% | 8.44 pp | -136 | 46 | -2.96 |
| BTC Daily | xgb | XGBoost | 816 | 318 | 498 | 38.97% | 35.42% | 35.62% | 11.03 pp | -180 | 46 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 578 | 280 | 298 | 48.44% | 46.67% | 47.50% | 1.56 pp | -18 | 54 | -0.33 |
| BTC Market Hours | nn | NN | 578 | 277 | 301 | 47.92% | 52.08% | 49.58% | 2.08 pp | -24 | 54 | -0.44 |
| BTC Market Hours | transformer | Transformer | 578 | 274 | 304 | 47.40% | 47.92% | 47.29% | 2.60 pp | -30 | 54 | -0.56 |
| BTC Market Hours | lstm | LSTM | 578 | 249 | 329 | 43.08% | 42.08% | 43.12% | 6.92 pp | -80 | 54 | -1.48 |
| BTC Market Hours | rf | RandomForest | 578 | 248 | 330 | 42.91% | 44.58% | 43.12% | 7.09 pp | -82 | 54 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 578 | 248 | 330 | 42.91% | 45.83% | 43.12% | 7.09 pp | -82 | 54 | -1.52 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 632 | 297 | 335 | 46.99% | 49.58% | 47.50% | 3.01 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 632 | 296 | 336 | 46.84% | 47.50% | 47.92% | 3.16 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 632 | 295 | 337 | 46.68% | 48.75% | 46.88% | 3.32 pp | -42 | 54 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 632 | 263 | 369 | 41.61% | 43.33% | 40.42% | 8.39 pp | -106 | 54 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 632 | 261 | 371 | 41.30% | 45.00% | 40.42% | 8.70 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 632 | 258 | 374 | 40.82% | 41.67% | 40.00% | 9.18 pp | -116 | 54 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
