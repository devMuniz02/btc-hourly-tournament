# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T02:19:03.702796+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1286 | 998 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1161 | 796 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 881 | 558 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 883 | 612 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 558 | 273 | 285 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 558 | 266 | 292 | 47.67% | 51.67% | 49.38% | 2.33 pp | -26 | 52 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 558 | 263 | 295 | 47.13% | 46.67% | 47.50% | 2.87 pp | -32 | 52 | -0.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 786 | 378 | 408 | 48.09% | 45.83% | 47.71% | 1.91 pp | -30 | 46 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 612 | 287 | 325 | 46.90% | 49.58% | 47.71% | 3.10 pp | -38 | 52 | -0.73 |
| BTC Market Hours Daily | nn | NN | 612 | 286 | 326 | 46.73% | 47.92% | 48.12% | 3.27 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 612 | 285 | 327 | 46.57% | 49.17% | 47.29% | 3.43 pp | -42 | 52 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 964 | 459 | 505 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| BTC Daily | transformer | Transformer | 786 | 365 | 421 | 46.44% | 40.42% | 46.25% | 3.56 pp | -56 | 46 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 786 | 363 | 423 | 46.18% | 44.17% | 44.79% | 3.82 pp | -60 | 46 | -1.30 |
| BTC Hourly | transformer | Transformer | 964 | 449 | 515 | 46.58% | 45.00% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 558 | 242 | 316 | 43.37% | 42.08% | 43.75% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 558 | 242 | 316 | 43.37% | 45.83% | 43.33% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 558 | 237 | 321 | 42.47% | 45.42% | 42.92% | 7.53 pp | -84 | 52 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 612 | 256 | 356 | 41.83% | 44.17% | 41.04% | 8.17 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 612 | 249 | 363 | 40.69% | 42.92% | 40.21% | 9.31 pp | -114 | 52 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 612 | 248 | 364 | 40.52% | 40.42% | 40.42% | 9.48 pp | -116 | 52 | -2.23 |
| BTC Hourly | nn | NN | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 786 | 331 | 455 | 42.11% | 33.75% | 39.79% | 7.89 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 786 | 328 | 458 | 41.73% | 37.92% | 41.46% | 8.27 pp | -130 | 46 | -2.83 |
| BTC Hourly | lstm | LSTM | 964 | 410 | 554 | 42.53% | 37.08% | 41.46% | 7.47 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 964 | 397 | 567 | 41.18% | 36.67% | 38.54% | 8.82 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 796 | 310 | 486 | 38.94% | 35.42% | 36.25% | 11.06 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 964 | 459 | 505 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 964 | 449 | 515 | 46.58% | 45.00% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| BTC Hourly | nn | NN | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 964 | 426 | 538 | 44.19% | 41.67% | 42.50% | 5.81 pp | -112 | 50 | -2.24 |
| BTC Hourly | lstm | LSTM | 964 | 410 | 554 | 42.53% | 37.08% | 41.46% | 7.47 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 964 | 397 | 567 | 41.18% | 36.67% | 38.54% | 8.82 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 786 | 378 | 408 | 48.09% | 45.83% | 47.71% | 1.91 pp | -30 | 46 | -0.65 |
| BTC Daily | transformer | Transformer | 786 | 365 | 421 | 46.44% | 40.42% | 46.25% | 3.56 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 786 | 363 | 423 | 46.18% | 44.17% | 44.79% | 3.82 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 786 | 331 | 455 | 42.11% | 33.75% | 39.79% | 7.89 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 786 | 328 | 458 | 41.73% | 37.92% | 41.46% | 8.27 pp | -130 | 46 | -2.83 |
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
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
