# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T23:22:03.079477+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1299 | 1011 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1175 | 810 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 22:00:00+00:00 | 906 | 572 | 333 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 22:00:00+00:00 | 908 | 626 | 280 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 572 | 278 | 294 | 48.60% | 47.50% | 47.92% | 1.40 pp | -16 | 53 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| BTC Market Hours | nn | NN | 572 | 272 | 300 | 47.55% | 51.67% | 49.17% | 2.45 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 800 | 386 | 414 | 48.25% | 47.08% | 47.71% | 1.75 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 572 | 268 | 304 | 46.85% | 46.67% | 46.67% | 3.15 pp | -36 | 53 | -0.68 |
| BTC Market Hours Daily | nn | NN | 626 | 292 | 334 | 46.65% | 47.92% | 48.12% | 3.35 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 626 | 292 | 334 | 46.65% | 48.75% | 47.50% | 3.35 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 626 | 291 | 335 | 46.49% | 48.75% | 47.08% | 3.51 pp | -44 | 53 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 977 | 466 | 511 | 47.70% | 50.42% | 47.08% | 2.30 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 800 | 372 | 428 | 46.50% | 45.42% | 45.00% | 3.50 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 800 | 370 | 430 | 46.25% | 39.17% | 45.83% | 3.75 pp | -60 | 46 | -1.30 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 977 | 453 | 524 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 572 | 248 | 324 | 43.36% | 42.92% | 43.75% | 6.64 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 572 | 246 | 326 | 43.01% | 45.00% | 43.54% | 6.99 pp | -80 | 53 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 572 | 246 | 326 | 43.01% | 46.25% | 43.33% | 6.99 pp | -80 | 53 | -1.51 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 626 | 260 | 366 | 41.53% | 42.50% | 40.62% | 8.47 pp | -106 | 53 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 626 | 258 | 368 | 41.21% | 43.75% | 40.83% | 8.79 pp | -110 | 53 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 626 | 255 | 371 | 40.73% | 41.25% | 40.21% | 9.27 pp | -116 | 53 | -2.19 |
| BTC Hourly | rf | RandomForest | 977 | 432 | 545 | 44.22% | 41.67% | 42.71% | 5.78 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 977 | 431 | 546 | 44.11% | 41.25% | 42.50% | 5.89 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 800 | 337 | 463 | 42.12% | 34.58% | 40.00% | 7.87 pp | -126 | 46 | -2.74 |
| BTC Hourly | lstm | LSTM | 977 | 416 | 561 | 42.58% | 37.92% | 41.04% | 7.42 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 800 | 334 | 466 | 41.75% | 37.08% | 41.25% | 8.25 pp | -132 | 46 | -2.87 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 977 | 404 | 573 | 41.35% | 35.83% | 39.17% | 8.65 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 810 | 315 | 495 | 38.89% | 34.58% | 35.62% | 11.11 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 977 | 466 | 511 | 47.70% | 50.42% | 47.08% | 2.30 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 977 | 453 | 524 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Hourly | rf | RandomForest | 977 | 432 | 545 | 44.22% | 41.67% | 42.71% | 5.78 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 977 | 431 | 546 | 44.11% | 41.25% | 42.50% | 5.89 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 977 | 416 | 561 | 42.58% | 37.92% | 41.04% | 7.42 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 977 | 404 | 573 | 41.35% | 35.83% | 39.17% | 8.65 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 800 | 386 | 414 | 48.25% | 47.08% | 47.71% | 1.75 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 800 | 372 | 428 | 46.50% | 45.42% | 45.00% | 3.50 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 800 | 370 | 430 | 46.25% | 39.17% | 45.83% | 3.75 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 800 | 337 | 463 | 42.12% | 34.58% | 40.00% | 7.87 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 800 | 334 | 466 | 41.75% | 37.08% | 41.25% | 8.25 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 810 | 315 | 495 | 38.89% | 34.58% | 35.62% | 11.11 pp | -180 | 46 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 572 | 278 | 294 | 48.60% | 47.50% | 47.92% | 1.40 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 572 | 272 | 300 | 47.55% | 51.67% | 49.17% | 2.45 pp | -28 | 53 | -0.53 |
| BTC Market Hours | transformer | Transformer | 572 | 268 | 304 | 46.85% | 46.67% | 46.67% | 3.15 pp | -36 | 53 | -0.68 |
| BTC Market Hours | lstm | LSTM | 572 | 248 | 324 | 43.36% | 42.92% | 43.75% | 6.64 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 572 | 246 | 326 | 43.01% | 45.00% | 43.54% | 6.99 pp | -80 | 53 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 572 | 246 | 326 | 43.01% | 46.25% | 43.33% | 6.99 pp | -80 | 53 | -1.51 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 626 | 292 | 334 | 46.65% | 47.92% | 48.12% | 3.35 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 626 | 292 | 334 | 46.65% | 48.75% | 47.50% | 3.35 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 626 | 291 | 335 | 46.49% | 48.75% | 47.08% | 3.51 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 626 | 260 | 366 | 41.53% | 42.50% | 40.62% | 8.47 pp | -106 | 53 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 626 | 258 | 368 | 41.21% | 43.75% | 40.83% | 8.79 pp | -110 | 53 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 626 | 255 | 371 | 40.73% | 41.25% | 40.21% | 9.27 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
