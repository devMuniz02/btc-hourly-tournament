# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T07:06:24.895022+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1240 | 952 | 288 | 0 |
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
| BTC Daily | transformer | Transformer | 740 | 352 | 388 | 47.57% | 45.83% | 49.38% | 2.43 pp | -36 | 44 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 918 | 439 | 479 | 47.82% | 49.58% | 47.50% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | nn | NN | 566 | 262 | 304 | 46.29% | 45.42% | 47.71% | 3.71 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 566 | 261 | 305 | 46.11% | 48.75% | 46.46% | 3.89 pp | -44 | 49 | -0.90 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 918 | 434 | 484 | 47.28% | 47.50% | 46.25% | 2.72 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Daily | nn | NN | 740 | 341 | 399 | 46.08% | 42.92% | 46.46% | 3.92 pp | -58 | 44 | -1.32 |
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
| BTC Hourly | rf | RandomForest | 918 | 409 | 509 | 44.55% | 44.17% | 44.38% | 5.45 pp | -100 | 48 | -2.08 |
| BTC Hourly | nn | NN | 918 | 408 | 510 | 44.44% | 43.33% | 42.29% | 5.56 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 566 | 230 | 336 | 40.64% | 39.58% | 40.83% | 9.36 pp | -106 | 49 | -2.16 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 566 | 227 | 339 | 40.11% | 40.83% | 39.38% | 9.89 pp | -112 | 49 | -2.29 |
| BTC Daily | lstm | LSTM | 740 | 318 | 422 | 42.97% | 37.08% | 41.04% | 7.03 pp | -104 | 44 | -2.36 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 740 | 312 | 428 | 42.16% | 39.17% | 42.50% | 7.84 pp | -116 | 44 | -2.64 |
| BTC Hourly | lstm | LSTM | 918 | 393 | 525 | 42.81% | 39.17% | 41.67% | 7.19 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 918 | 383 | 535 | 41.72% | 39.17% | 40.00% | 8.28 pp | -152 | 48 | -3.17 |
| BTC Daily | xgb | XGBoost | 750 | 296 | 454 | 39.47% | 36.25% | 37.71% | 10.53 pp | -158 | 44 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 918 | 439 | 479 | 47.82% | 49.58% | 47.50% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Hourly | transformer | Transformer | 918 | 434 | 484 | 47.28% | 47.50% | 46.25% | 2.72 pp | -50 | 48 | -1.04 |
| BTC Hourly | rf | RandomForest | 918 | 409 | 509 | 44.55% | 44.17% | 44.38% | 5.45 pp | -100 | 48 | -2.08 |
| BTC Hourly | nn | NN | 918 | 408 | 510 | 44.44% | 43.33% | 42.29% | 5.56 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 918 | 393 | 525 | 42.81% | 39.17% | 41.67% | 7.19 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 918 | 383 | 535 | 41.72% | 39.17% | 40.00% | 8.28 pp | -152 | 48 | -3.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 740 | 358 | 382 | 48.38% | 46.67% | 48.33% | 1.62 pp | -24 | 44 | -0.55 |
| BTC Daily | transformer | Transformer | 740 | 352 | 388 | 47.57% | 45.83% | 49.38% | 2.43 pp | -36 | 44 | -0.82 |
| BTC Daily | nn | NN | 740 | 341 | 399 | 46.08% | 42.92% | 46.46% | 3.92 pp | -58 | 44 | -1.32 |
| BTC Daily | lstm | LSTM | 740 | 318 | 422 | 42.97% | 37.08% | 41.04% | 7.03 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 740 | 312 | 428 | 42.16% | 39.17% | 42.50% | 7.84 pp | -116 | 44 | -2.64 |
| BTC Daily | xgb | XGBoost | 750 | 296 | 454 | 39.47% | 36.25% | 37.71% | 10.53 pp | -158 | 44 | -3.59 |

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
