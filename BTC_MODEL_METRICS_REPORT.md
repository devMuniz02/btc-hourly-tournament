# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T03:03:46.156905+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1145 | 780 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 852 | 542 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 854 | 596 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 274 | 268 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| BTC Market Hours | nn | NN | 542 | 267 | 275 | 49.26% | 51.25% | 50.83% | 0.74 pp | -8 | 51 | -0.16 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 596 | 284 | 312 | 47.65% | 46.67% | 48.12% | 2.35 pp | -28 | 51 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 770 | 371 | 399 | 48.18% | 46.25% | 47.92% | 1.82 pp | -28 | 45 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 596 | 280 | 316 | 46.98% | 47.92% | 46.88% | 3.02 pp | -36 | 51 | -0.71 |
| BTC Market Hours | transformer | Transformer | 542 | 252 | 290 | 46.49% | 45.83% | 47.08% | 3.51 pp | -38 | 51 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 948 | 453 | 495 | 47.78% | 49.58% | 47.08% | 2.22 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 596 | 275 | 321 | 46.14% | 47.92% | 47.29% | 3.86 pp | -46 | 51 | -0.90 |
| BTC Daily | transformer | Transformer | 770 | 359 | 411 | 46.62% | 41.25% | 46.88% | 3.38 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 542 | 239 | 303 | 44.10% | 46.25% | 43.96% | 5.90 pp | -64 | 51 | -1.25 |
| BTC Daily | nn | NN | 770 | 356 | 414 | 46.23% | 44.58% | 45.83% | 3.77 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 948 | 442 | 506 | 46.62% | 45.42% | 44.58% | 3.38 pp | -64 | 49 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 596 | 262 | 334 | 43.96% | 46.25% | 43.54% | 6.04 pp | -72 | 51 | -1.41 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| BTC Market Hours | lstm | LSTM | 542 | 226 | 316 | 41.70% | 37.08% | 41.88% | 8.30 pp | -90 | 51 | -1.76 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 542 | 224 | 318 | 41.33% | 42.92% | 41.67% | 8.67 pp | -94 | 51 | -1.84 |
| BTC Market Hours Daily | xgb | XGBoost | 596 | 249 | 347 | 41.78% | 42.50% | 41.46% | 8.22 pp | -98 | 51 | -1.92 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 948 | 421 | 527 | 44.41% | 44.17% | 43.75% | 5.59 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Hourly | nn | NN | 948 | 420 | 528 | 44.30% | 42.08% | 42.92% | 5.70 pp | -108 | 49 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 596 | 236 | 360 | 39.60% | 35.42% | 39.17% | 10.40 pp | -124 | 51 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 770 | 325 | 445 | 42.21% | 35.83% | 40.21% | 7.79 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 770 | 322 | 448 | 41.82% | 38.33% | 42.08% | 8.18 pp | -126 | 45 | -2.80 |
| BTC Hourly | lstm | LSTM | 948 | 405 | 543 | 42.72% | 36.25% | 42.29% | 7.28 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 948 | 396 | 552 | 41.77% | 39.58% | 40.42% | 8.23 pp | -156 | 49 | -3.18 |
| BTC Daily | xgb | XGBoost | 780 | 304 | 476 | 38.97% | 34.58% | 36.25% | 11.03 pp | -172 | 45 | -3.82 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 770 | 371 | 399 | 48.18% | 46.25% | 47.92% | 1.82 pp | -28 | 45 | -0.62 |
| BTC Daily | transformer | Transformer | 770 | 359 | 411 | 46.62% | 41.25% | 46.88% | 3.38 pp | -52 | 45 | -1.16 |
| BTC Daily | nn | NN | 770 | 356 | 414 | 46.23% | 44.58% | 45.83% | 3.77 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 770 | 325 | 445 | 42.21% | 35.83% | 40.21% | 7.79 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 770 | 322 | 448 | 41.82% | 38.33% | 42.08% | 8.18 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 780 | 304 | 476 | 38.97% | 34.58% | 36.25% | 11.03 pp | -172 | 45 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 542 | 274 | 268 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| BTC Market Hours | nn | NN | 542 | 267 | 275 | 49.26% | 51.25% | 50.83% | 0.74 pp | -8 | 51 | -0.16 |
| BTC Market Hours | transformer | Transformer | 542 | 252 | 290 | 46.49% | 45.83% | 47.08% | 3.51 pp | -38 | 51 | -0.75 |
| BTC Market Hours | rf | RandomForest | 542 | 239 | 303 | 44.10% | 46.25% | 43.96% | 5.90 pp | -64 | 51 | -1.25 |
| BTC Market Hours | lstm | LSTM | 542 | 226 | 316 | 41.70% | 37.08% | 41.88% | 8.30 pp | -90 | 51 | -1.76 |
| BTC Market Hours | xgb | XGBoost | 542 | 224 | 318 | 41.33% | 42.92% | 41.67% | 8.67 pp | -94 | 51 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 596 | 284 | 312 | 47.65% | 46.67% | 48.12% | 2.35 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 596 | 280 | 316 | 46.98% | 47.92% | 46.88% | 3.02 pp | -36 | 51 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 596 | 275 | 321 | 46.14% | 47.92% | 47.29% | 3.86 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 596 | 262 | 334 | 43.96% | 46.25% | 43.54% | 6.04 pp | -72 | 51 | -1.41 |
| BTC Market Hours Daily | xgb | XGBoost | 596 | 249 | 347 | 41.78% | 42.50% | 41.46% | 8.22 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 596 | 236 | 360 | 39.60% | 35.42% | 39.17% | 10.40 pp | -124 | 51 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
