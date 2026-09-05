# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T06:58:24.532268+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1239 | 951 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1115 | 750 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 796 | 512 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 798 | 566 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 512 | 248 | 264 | 48.44% | 45.83% | 48.54% | 1.56 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 512 | 246 | 266 | 48.05% | 47.08% | 48.33% | 1.95 pp | -20 | 49 | -0.41 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 566 | 270 | 296 | 47.70% | 51.67% | 48.96% | 2.30 pp | -26 | 49 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 740 | 358 | 382 | 48.38% | 46.67% | 48.33% | 1.62 pp | -24 | 44 | -0.55 |
| BTC Market Hours | nn | NN | 512 | 241 | 271 | 47.07% | 49.58% | 48.12% | 2.93 pp | -30 | 49 | -0.61 |
| BTC Daily | transformer | Transformer | 740 | 353 | 387 | 47.70% | 46.25% | 49.58% | 2.30 pp | -34 | 44 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 917 | 439 | 478 | 47.87% | 49.58% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | nn | NN | 566 | 262 | 304 | 46.29% | 45.42% | 47.71% | 3.71 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 566 | 261 | 305 | 46.11% | 48.75% | 46.46% | 3.89 pp | -44 | 49 | -0.90 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 917 | 434 | 483 | 47.33% | 47.92% | 46.46% | 2.67 pp | -49 | 48 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Daily | nn | NN | 740 | 342 | 398 | 46.22% | 43.33% | 46.67% | 3.78 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 512 | 222 | 290 | 43.36% | 42.50% | 43.54% | 6.64 pp | -68 | 49 | -1.39 |
| BTC Market Hours | rf | RandomForest | 512 | 221 | 291 | 43.16% | 44.58% | 43.54% | 6.84 pp | -70 | 49 | -1.43 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 512 | 210 | 302 | 41.02% | 42.50% | 41.67% | 8.98 pp | -92 | 49 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 566 | 236 | 330 | 41.70% | 42.50% | 40.83% | 8.30 pp | -94 | 49 | -1.92 |
| BTC Hourly | rf | RandomForest | 917 | 409 | 508 | 44.60% | 44.17% | 44.38% | 5.40 pp | -99 | 48 | -2.06 |
| BTC Hourly | nn | NN | 917 | 408 | 509 | 44.49% | 43.33% | 42.29% | 5.51 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 566 | 230 | 336 | 40.64% | 39.58% | 40.83% | 9.36 pp | -106 | 49 | -2.16 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 566 | 227 | 339 | 40.11% | 40.83% | 39.38% | 9.89 pp | -112 | 49 | -2.29 |
| BTC Daily | lstm | LSTM | 740 | 318 | 422 | 42.97% | 37.08% | 41.04% | 7.03 pp | -104 | 44 | -2.36 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 740 | 313 | 427 | 42.30% | 39.58% | 42.71% | 7.70 pp | -114 | 44 | -2.59 |
| BTC Hourly | lstm | LSTM | 917 | 393 | 524 | 42.86% | 39.17% | 41.67% | 7.14 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 917 | 383 | 534 | 41.77% | 39.58% | 40.00% | 8.23 pp | -151 | 48 | -3.15 |
| BTC Daily | xgb | XGBoost | 750 | 297 | 453 | 39.60% | 36.67% | 37.92% | 10.40 pp | -156 | 44 | -3.55 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 917 | 439 | 478 | 47.87% | 49.58% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 917 | 434 | 483 | 47.33% | 47.92% | 46.46% | 2.67 pp | -49 | 48 | -1.02 |
| BTC Hourly | rf | RandomForest | 917 | 409 | 508 | 44.60% | 44.17% | 44.38% | 5.40 pp | -99 | 48 | -2.06 |
| BTC Hourly | nn | NN | 917 | 408 | 509 | 44.49% | 43.33% | 42.29% | 5.51 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 917 | 393 | 524 | 42.86% | 39.17% | 41.67% | 7.14 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 917 | 383 | 534 | 41.77% | 39.58% | 40.00% | 8.23 pp | -151 | 48 | -3.15 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 740 | 358 | 382 | 48.38% | 46.67% | 48.33% | 1.62 pp | -24 | 44 | -0.55 |
| BTC Daily | transformer | Transformer | 740 | 353 | 387 | 47.70% | 46.25% | 49.58% | 2.30 pp | -34 | 44 | -0.77 |
| BTC Daily | nn | NN | 740 | 342 | 398 | 46.22% | 43.33% | 46.67% | 3.78 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 740 | 318 | 422 | 42.97% | 37.08% | 41.04% | 7.03 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 740 | 313 | 427 | 42.30% | 39.58% | 42.71% | 7.70 pp | -114 | 44 | -2.59 |
| BTC Daily | xgb | XGBoost | 750 | 297 | 453 | 39.60% | 36.67% | 37.92% | 10.40 pp | -156 | 44 | -3.55 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 512 | 248 | 264 | 48.44% | 45.83% | 48.54% | 1.56 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 512 | 246 | 266 | 48.05% | 47.08% | 48.33% | 1.95 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 512 | 241 | 271 | 47.07% | 49.58% | 48.12% | 2.93 pp | -30 | 49 | -0.61 |
| BTC Market Hours | lstm | LSTM | 512 | 222 | 290 | 43.36% | 42.50% | 43.54% | 6.64 pp | -68 | 49 | -1.39 |
| BTC Market Hours | rf | RandomForest | 512 | 221 | 291 | 43.16% | 44.58% | 43.54% | 6.84 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 512 | 210 | 302 | 41.02% | 42.50% | 41.67% | 8.98 pp | -92 | 49 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 566 | 270 | 296 | 47.70% | 51.67% | 48.96% | 2.30 pp | -26 | 49 | -0.53 |
| BTC Market Hours Daily | nn | NN | 566 | 262 | 304 | 46.29% | 45.42% | 47.71% | 3.71 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 566 | 261 | 305 | 46.11% | 48.75% | 46.46% | 3.89 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 566 | 236 | 330 | 41.70% | 42.50% | 40.83% | 8.30 pp | -94 | 49 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 566 | 230 | 336 | 40.64% | 39.58% | 40.83% | 9.36 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 566 | 227 | 339 | 40.11% | 40.83% | 39.38% | 9.89 pp | -112 | 49 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
