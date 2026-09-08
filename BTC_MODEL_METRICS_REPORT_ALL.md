# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T00:59:47.002080+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 880 | 558 | 321 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 881 | 611 | 268 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 558 | 273 | 285 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 558 | 266 | 292 | 47.67% | 51.67% | 49.38% | 2.33 pp | -26 | 52 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 786 | 379 | 407 | 48.22% | 46.25% | 47.92% | 1.78 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 558 | 263 | 295 | 47.13% | 46.67% | 47.50% | 2.87 pp | -32 | 52 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 611 | 287 | 324 | 46.97% | 50.00% | 47.92% | 3.03 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 611 | 286 | 325 | 46.81% | 48.33% | 48.33% | 3.19 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 611 | 285 | 326 | 46.64% | 49.58% | 47.50% | 3.36 pp | -41 | 52 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 963 | 458 | 505 | 47.56% | 49.58% | 46.88% | 2.44 pp | -47 | 50 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 786 | 365 | 421 | 46.44% | 40.42% | 46.25% | 3.56 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 786 | 363 | 423 | 46.18% | 44.17% | 44.79% | 3.82 pp | -60 | 46 | -1.30 |
| BTC Hourly | transformer | Transformer | 963 | 448 | 515 | 46.52% | 45.00% | 43.75% | 3.48 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 558 | 242 | 316 | 43.37% | 42.08% | 43.75% | 6.63 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 558 | 242 | 316 | 43.37% | 45.83% | 43.33% | 6.63 pp | -74 | 52 | -1.42 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 558 | 237 | 321 | 42.47% | 45.42% | 42.92% | 7.53 pp | -84 | 52 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 611 | 256 | 355 | 41.90% | 44.17% | 41.04% | 8.10 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 611 | 248 | 363 | 40.59% | 40.42% | 40.42% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 611 | 248 | 363 | 40.59% | 42.50% | 40.00% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Hourly | nn | NN | 963 | 426 | 537 | 44.24% | 42.08% | 42.50% | 5.76 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 963 | 426 | 537 | 44.24% | 42.08% | 42.71% | 5.76 pp | -111 | 50 | -2.22 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
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
| BTC Market Hours Daily | transformer | Transformer | 611 | 287 | 324 | 46.97% | 50.00% | 47.92% | 3.03 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 611 | 286 | 325 | 46.81% | 48.33% | 48.33% | 3.19 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 611 | 285 | 326 | 46.64% | 49.58% | 47.50% | 3.36 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 611 | 256 | 355 | 41.90% | 44.17% | 41.04% | 8.10 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 611 | 248 | 363 | 40.59% | 40.42% | 40.42% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 611 | 248 | 363 | 40.59% | 42.50% | 40.00% | 9.41 pp | -115 | 52 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
