# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T03:33:25.531576+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1270 | 982 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1146 | 781 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 853 | 543 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 855 | 597 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 187 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T15:00:00+00:00 | 188 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 543 | 275 | 268 | 50.64% | 47.50% | 50.42% | 0.64 pp | 7 | 51 | 0.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | nn | NN | 543 | 268 | 275 | 49.36% | 51.67% | 51.04% | 0.64 pp | -7 | 51 | -0.14 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | nn | NN | 597 | 285 | 312 | 47.74% | 47.08% | 48.12% | 2.26 pp | -27 | 51 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 771 | 372 | 399 | 48.25% | 46.25% | 47.92% | 1.75 pp | -27 | 45 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 597 | 281 | 316 | 47.07% | 47.92% | 46.88% | 2.93 pp | -35 | 51 | -0.69 |
| BTC Market Hours | transformer | Transformer | 543 | 252 | 291 | 46.41% | 45.83% | 47.08% | 3.59 pp | -39 | 51 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 948 | 453 | 495 | 47.78% | 49.58% | 47.08% | 2.22 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 597 | 276 | 321 | 46.23% | 47.92% | 47.50% | 3.77 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| BTC Daily | transformer | Transformer | 771 | 360 | 411 | 46.69% | 41.67% | 47.08% | 3.31 pp | -51 | 45 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| BTC Market Hours | rf | RandomForest | 543 | 240 | 303 | 44.20% | 46.67% | 44.17% | 5.80 pp | -63 | 51 | -1.24 |
| BTC Daily | nn | NN | 771 | 357 | 414 | 46.30% | 44.58% | 45.83% | 3.70 pp | -57 | 45 | -1.27 |
| BTC Hourly | transformer | Transformer | 948 | 442 | 506 | 46.62% | 45.42% | 44.58% | 3.38 pp | -64 | 49 | -1.31 |
| Consolidated Hourly | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 597 | 263 | 334 | 44.05% | 46.25% | 43.54% | 5.95 pp | -71 | 51 | -1.39 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 543 | 226 | 317 | 41.62% | 37.08% | 41.88% | 8.38 pp | -91 | 51 | -1.78 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 543 | 225 | 318 | 41.44% | 42.92% | 41.67% | 8.56 pp | -93 | 51 | -1.82 |
| BTC Market Hours Daily | xgb | XGBoost | 597 | 250 | 347 | 41.88% | 42.50% | 41.46% | 8.12 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Hourly | rf | RandomForest | 948 | 421 | 527 | 44.41% | 44.17% | 43.75% | 5.59 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Hourly | nn | NN | 948 | 420 | 528 | 44.30% | 42.08% | 42.92% | 5.70 pp | -108 | 49 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 597 | 237 | 360 | 39.70% | 35.83% | 39.38% | 10.30 pp | -123 | 51 | -2.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 771 | 325 | 446 | 42.15% | 35.42% | 40.21% | 7.85 pp | -121 | 45 | -2.69 |
| BTC Daily | rf | RandomForest | 771 | 323 | 448 | 41.89% | 38.33% | 42.29% | 8.11 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 948 | 405 | 543 | 42.72% | 36.25% | 42.29% | 7.28 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 948 | 396 | 552 | 41.77% | 39.58% | 40.42% | 8.23 pp | -156 | 49 | -3.18 |
| BTC Daily | xgb | XGBoost | 781 | 305 | 476 | 39.05% | 34.58% | 36.46% | 10.95 pp | -171 | 45 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 948 | 453 | 495 | 47.78% | 49.58% | 47.08% | 2.22 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 948 | 442 | 506 | 46.62% | 45.42% | 44.58% | 3.38 pp | -64 | 49 | -1.31 |
| BTC Hourly | rf | RandomForest | 948 | 421 | 527 | 44.41% | 44.17% | 43.75% | 5.59 pp | -106 | 49 | -2.16 |
| BTC Hourly | nn | NN | 948 | 420 | 528 | 44.30% | 42.08% | 42.92% | 5.70 pp | -108 | 49 | -2.20 |
| BTC Hourly | lstm | LSTM | 948 | 405 | 543 | 42.72% | 36.25% | 42.29% | 7.28 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 948 | 396 | 552 | 41.77% | 39.58% | 40.42% | 8.23 pp | -156 | 49 | -3.18 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 771 | 372 | 399 | 48.25% | 46.25% | 47.92% | 1.75 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 771 | 360 | 411 | 46.69% | 41.67% | 47.08% | 3.31 pp | -51 | 45 | -1.13 |
| BTC Daily | nn | NN | 771 | 357 | 414 | 46.30% | 44.58% | 45.83% | 3.70 pp | -57 | 45 | -1.27 |
| BTC Daily | lstm | LSTM | 771 | 325 | 446 | 42.15% | 35.42% | 40.21% | 7.85 pp | -121 | 45 | -2.69 |
| BTC Daily | rf | RandomForest | 771 | 323 | 448 | 41.89% | 38.33% | 42.29% | 8.11 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 781 | 305 | 476 | 39.05% | 34.58% | 36.46% | 10.95 pp | -171 | 45 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 543 | 275 | 268 | 50.64% | 47.50% | 50.42% | 0.64 pp | 7 | 51 | 0.14 |
| BTC Market Hours | nn | NN | 543 | 268 | 275 | 49.36% | 51.67% | 51.04% | 0.64 pp | -7 | 51 | -0.14 |
| BTC Market Hours | transformer | Transformer | 543 | 252 | 291 | 46.41% | 45.83% | 47.08% | 3.59 pp | -39 | 51 | -0.76 |
| BTC Market Hours | rf | RandomForest | 543 | 240 | 303 | 44.20% | 46.67% | 44.17% | 5.80 pp | -63 | 51 | -1.24 |
| BTC Market Hours | lstm | LSTM | 543 | 226 | 317 | 41.62% | 37.08% | 41.88% | 8.38 pp | -91 | 51 | -1.78 |
| BTC Market Hours | xgb | XGBoost | 543 | 225 | 318 | 41.44% | 42.92% | 41.67% | 8.56 pp | -93 | 51 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 597 | 285 | 312 | 47.74% | 47.08% | 48.12% | 2.26 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 597 | 281 | 316 | 47.07% | 47.92% | 46.88% | 2.93 pp | -35 | 51 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 597 | 276 | 321 | 46.23% | 47.92% | 47.50% | 3.77 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 597 | 263 | 334 | 44.05% | 46.25% | 43.54% | 5.95 pp | -71 | 51 | -1.39 |
| BTC Market Hours Daily | xgb | XGBoost | 597 | 250 | 347 | 41.88% | 42.50% | 41.46% | 8.12 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 597 | 237 | 360 | 39.70% | 35.83% | 39.38% | 10.30 pp | -123 | 51 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 92 | 95 | 49.20% | 49.20% | 49.20% | 0.80 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 85 | 102 | 45.45% | 45.45% | 45.45% | 4.55 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
