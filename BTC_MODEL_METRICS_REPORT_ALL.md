# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T11:09:59.615940+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1194 | 906 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1070 | 705 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 712 | 467 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 714 | 521 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 467 | 227 | 240 | 48.61% | 44.58% | 48.61% | 1.39 pp | -13 | 45 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 695 | 339 | 356 | 48.78% | 45.83% | 49.17% | 1.22 pp | -17 | 42 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | transformer | Transformer | 695 | 334 | 361 | 48.06% | 46.25% | 49.38% | 1.94 pp | -27 | 42 | -0.64 |
| BTC Market Hours | nn | NN | 467 | 219 | 248 | 46.90% | 47.92% | 46.90% | 3.10 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 467 | 218 | 249 | 46.68% | 41.25% | 46.68% | 3.32 pp | -31 | 45 | -0.69 |
| Consolidated Hourly | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 521 | 239 | 282 | 45.87% | 46.67% | 46.46% | 4.13 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 872 | 413 | 459 | 47.36% | 47.92% | 47.50% | 2.64 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 521 | 238 | 283 | 45.68% | 47.08% | 46.46% | 4.32 pp | -45 | 45 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 521 | 237 | 284 | 45.49% | 42.50% | 46.04% | 4.51 pp | -47 | 45 | -1.04 |
| BTC Hourly | transformer | Transformer | 872 | 411 | 461 | 47.13% | 48.33% | 47.29% | 2.87 pp | -50 | 46 | -1.09 |
| Consolidated Hourly | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 695 | 323 | 372 | 46.47% | 42.92% | 48.54% | 3.53 pp | -49 | 42 | -1.17 |
| BTC Market Hours | rf | RandomForest | 467 | 201 | 266 | 43.04% | 42.92% | 43.04% | 6.96 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 467 | 200 | 267 | 42.83% | 40.83% | 42.83% | 7.17 pp | -67 | 45 | -1.49 |
| Consolidated Hourly | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 872 | 392 | 480 | 44.95% | 45.83% | 43.75% | 5.05 pp | -88 | 46 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 467 | 190 | 277 | 40.69% | 39.58% | 40.69% | 9.31 pp | -87 | 45 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 521 | 215 | 306 | 41.27% | 40.83% | 41.25% | 8.73 pp | -91 | 45 | -2.02 |
| BTC Hourly | rf | RandomForest | 872 | 389 | 483 | 44.61% | 45.00% | 44.17% | 5.39 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 695 | 302 | 393 | 43.45% | 38.33% | 42.29% | 6.55 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 521 | 208 | 313 | 39.92% | 37.50% | 40.83% | 10.08 pp | -105 | 45 | -2.33 |
| BTC Daily | rf | RandomForest | 695 | 298 | 397 | 42.88% | 40.42% | 43.33% | 7.12 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 521 | 206 | 315 | 39.54% | 37.08% | 39.17% | 10.46 pp | -109 | 45 | -2.42 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 872 | 371 | 501 | 42.55% | 38.75% | 41.67% | 7.45 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 872 | 369 | 503 | 42.32% | 41.25% | 43.12% | 7.68 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 705 | 279 | 426 | 39.57% | 35.83% | 39.38% | 10.43 pp | -147 | 42 | -3.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 872 | 413 | 459 | 47.36% | 47.92% | 47.50% | 2.64 pp | -46 | 46 | -1.00 |
| BTC Hourly | transformer | Transformer | 872 | 411 | 461 | 47.13% | 48.33% | 47.29% | 2.87 pp | -50 | 46 | -1.09 |
| BTC Hourly | nn | NN | 872 | 392 | 480 | 44.95% | 45.83% | 43.75% | 5.05 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 872 | 389 | 483 | 44.61% | 45.00% | 44.17% | 5.39 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 872 | 371 | 501 | 42.55% | 38.75% | 41.67% | 7.45 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 872 | 369 | 503 | 42.32% | 41.25% | 43.12% | 7.68 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 695 | 339 | 356 | 48.78% | 45.83% | 49.17% | 1.22 pp | -17 | 42 | -0.40 |
| BTC Daily | transformer | Transformer | 695 | 334 | 361 | 48.06% | 46.25% | 49.38% | 1.94 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 695 | 323 | 372 | 46.47% | 42.92% | 48.54% | 3.53 pp | -49 | 42 | -1.17 |
| BTC Daily | lstm | LSTM | 695 | 302 | 393 | 43.45% | 38.33% | 42.29% | 6.55 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 695 | 298 | 397 | 42.88% | 40.42% | 43.33% | 7.12 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 705 | 279 | 426 | 39.57% | 35.83% | 39.38% | 10.43 pp | -147 | 42 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 467 | 227 | 240 | 48.61% | 44.58% | 48.61% | 1.39 pp | -13 | 45 | -0.29 |
| BTC Market Hours | nn | NN | 467 | 219 | 248 | 46.90% | 47.92% | 46.90% | 3.10 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 467 | 218 | 249 | 46.68% | 41.25% | 46.68% | 3.32 pp | -31 | 45 | -0.69 |
| BTC Market Hours | rf | RandomForest | 467 | 201 | 266 | 43.04% | 42.92% | 43.04% | 6.96 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 467 | 200 | 267 | 42.83% | 40.83% | 42.83% | 7.17 pp | -67 | 45 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 467 | 190 | 277 | 40.69% | 39.58% | 40.69% | 9.31 pp | -87 | 45 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 521 | 239 | 282 | 45.87% | 46.67% | 46.46% | 4.13 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 521 | 238 | 283 | 45.68% | 47.08% | 46.46% | 4.32 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | nn | NN | 521 | 237 | 284 | 45.49% | 42.50% | 46.04% | 4.51 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | rf | RandomForest | 521 | 215 | 306 | 41.27% | 40.83% | 41.25% | 8.73 pp | -91 | 45 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 521 | 208 | 313 | 39.92% | 37.50% | 40.83% | 10.08 pp | -105 | 45 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 521 | 206 | 315 | 39.54% | 37.08% | 39.17% | 10.46 pp | -109 | 45 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
