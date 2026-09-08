# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T01:29:39.414599+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1285 | 997 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1161 | 796 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 881 | 558 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 883 | 612 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 203 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 558 | 273 | 285 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 558 | 266 | 292 | 47.67% | 51.67% | 49.38% | 2.33 pp | -26 | 52 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 786 | 379 | 407 | 48.22% | 46.25% | 47.92% | 1.78 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 558 | 263 | 295 | 47.13% | 46.67% | 47.50% | 2.87 pp | -32 | 52 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 612 | 287 | 325 | 46.90% | 49.58% | 47.71% | 3.10 pp | -38 | 52 | -0.73 |
| BTC Market Hours Daily | nn | NN | 612 | 286 | 326 | 46.73% | 47.92% | 48.12% | 3.27 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 612 | 285 | 327 | 46.57% | 49.17% | 47.29% | 3.43 pp | -42 | 52 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 963 | 458 | 505 | 47.56% | 49.58% | 46.88% | 2.44 pp | -47 | 50 | -0.94 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 30 | 36 | 45.45% | 45.45% | 45.45% | 4.55 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 786 | 365 | 421 | 46.44% | 40.42% | 46.25% | 3.56 pp | -56 | 46 | -1.22 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 786 | 363 | 423 | 46.18% | 44.17% | 44.79% | 3.82 pp | -60 | 46 | -1.30 |
| BTC Hourly | transformer | Transformer | 963 | 448 | 515 | 46.52% | 45.00% | 43.75% | 3.48 pp | -67 | 50 | -1.34 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 558 | 242 | 316 | 43.37% | 42.08% | 43.75% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 558 | 242 | 316 | 43.37% | 45.83% | 43.33% | 6.63 pp | -74 | 52 | -1.42 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 558 | 237 | 321 | 42.47% | 45.42% | 42.92% | 7.53 pp | -84 | 52 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 612 | 256 | 356 | 41.83% | 44.17% | 41.04% | 8.17 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 612 | 249 | 363 | 40.69% | 42.92% | 40.21% | 9.31 pp | -114 | 52 | -2.19 |
| BTC Hourly | nn | NN | 963 | 426 | 537 | 44.24% | 42.08% | 42.50% | 5.76 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 963 | 426 | 537 | 44.24% | 42.08% | 42.71% | 5.76 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 612 | 248 | 364 | 40.52% | 40.42% | 40.42% | 9.48 pp | -116 | 52 | -2.23 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 786 | 330 | 456 | 41.98% | 33.33% | 39.58% | 8.02 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 786 | 329 | 457 | 41.86% | 38.33% | 41.67% | 8.14 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 963 | 410 | 553 | 42.58% | 37.08% | 41.67% | 7.42 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 963 | 397 | 566 | 41.23% | 36.67% | 38.75% | 8.77 pp | -169 | 50 | -3.38 |
| BTC Daily | xgb | XGBoost | 796 | 310 | 486 | 38.94% | 35.42% | 36.25% | 11.06 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 963 | 458 | 505 | 47.56% | 49.58% | 46.88% | 2.44 pp | -47 | 50 | -0.94 |
| BTC Hourly | transformer | Transformer | 963 | 448 | 515 | 46.52% | 45.00% | 43.75% | 3.48 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 963 | 426 | 537 | 44.24% | 42.08% | 42.50% | 5.76 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 963 | 426 | 537 | 44.24% | 42.08% | 42.71% | 5.76 pp | -111 | 50 | -2.22 |
| BTC Hourly | lstm | LSTM | 963 | 410 | 553 | 42.58% | 37.08% | 41.67% | 7.42 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 963 | 397 | 566 | 41.23% | 36.67% | 38.75% | 8.77 pp | -169 | 50 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 786 | 379 | 407 | 48.22% | 46.25% | 47.92% | 1.78 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 786 | 365 | 421 | 46.44% | 40.42% | 46.25% | 3.56 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 786 | 363 | 423 | 46.18% | 44.17% | 44.79% | 3.82 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 786 | 330 | 456 | 41.98% | 33.33% | 39.58% | 8.02 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 786 | 329 | 457 | 41.86% | 38.33% | 41.67% | 8.14 pp | -128 | 46 | -2.78 |
| BTC Daily | xgb | XGBoost | 796 | 310 | 486 | 38.94% | 35.42% | 36.25% | 11.06 pp | -176 | 46 | -3.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 558 | 273 | 285 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 558 | 266 | 292 | 47.67% | 51.67% | 49.38% | 2.33 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 558 | 263 | 295 | 47.13% | 46.67% | 47.50% | 2.87 pp | -32 | 52 | -0.62 |
| BTC Market Hours | lstm | LSTM | 558 | 242 | 316 | 43.37% | 42.08% | 43.75% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 558 | 242 | 316 | 43.37% | 45.83% | 43.33% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 558 | 237 | 321 | 42.47% | 45.42% | 42.92% | 7.53 pp | -84 | 52 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 612 | 287 | 325 | 46.90% | 49.58% | 47.71% | 3.10 pp | -38 | 52 | -0.73 |
| BTC Market Hours Daily | nn | NN | 612 | 286 | 326 | 46.73% | 47.92% | 48.12% | 3.27 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 612 | 285 | 327 | 46.57% | 49.17% | 47.29% | 3.43 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 612 | 256 | 356 | 41.83% | 44.17% | 41.04% | 8.17 pp | -100 | 52 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 612 | 249 | 363 | 40.69% | 42.92% | 40.21% | 9.31 pp | -114 | 52 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 612 | 248 | 364 | 40.52% | 40.42% | 40.42% | 9.48 pp | -116 | 52 | -2.23 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 30 | 36 | 45.45% | 45.45% | 45.45% | 4.55 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
