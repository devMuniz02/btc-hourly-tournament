# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T10:27:16.088259+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1069 | 704 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 711 | 466 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 713 | 520 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 117 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 117 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 19 | 98 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 19 | 98 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 466 | 226 | 240 | 48.50% | 44.17% | 48.50% | 1.50 pp | -14 | 45 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 694 | 338 | 356 | 48.70% | 45.42% | 48.96% | 1.30 pp | -18 | 42 | -0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | transformer | Transformer | 694 | 333 | 361 | 47.98% | 45.83% | 49.38% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Market Hours | nn | NN | 466 | 218 | 248 | 46.78% | 47.50% | 46.78% | 3.22 pp | -30 | 45 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 466 | 217 | 249 | 46.57% | 40.83% | 46.57% | 3.43 pp | -32 | 45 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 520 | 238 | 282 | 45.77% | 46.25% | 46.25% | 4.23 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 872 | 413 | 459 | 47.36% | 47.92% | 47.50% | 2.64 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | nn | NN | 520 | 237 | 283 | 45.58% | 42.50% | 46.04% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 520 | 237 | 283 | 45.58% | 47.08% | 46.25% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Hourly | transformer | Transformer | 872 | 411 | 461 | 47.13% | 48.33% | 47.29% | 2.87 pp | -50 | 46 | -1.09 |
| Consolidated Hourly | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 694 | 322 | 372 | 46.40% | 42.50% | 48.33% | 3.60 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 466 | 200 | 266 | 42.92% | 42.92% | 42.92% | 7.08 pp | -66 | 45 | -1.47 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 466 | 199 | 267 | 42.70% | 40.42% | 42.70% | 7.30 pp | -68 | 45 | -1.51 |
| Consolidated Hourly | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |
| BTC Hourly | nn | NN | 872 | 392 | 480 | 44.95% | 45.83% | 43.75% | 5.05 pp | -88 | 46 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 466 | 189 | 277 | 40.56% | 39.58% | 40.56% | 9.44 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 520 | 215 | 305 | 41.35% | 41.25% | 41.25% | 8.65 pp | -90 | 45 | -2.00 |
| BTC Hourly | rf | RandomForest | 872 | 389 | 483 | 44.61% | 45.00% | 44.17% | 5.39 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 694 | 302 | 392 | 43.52% | 38.33% | 42.50% | 6.48 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 520 | 208 | 312 | 40.00% | 37.92% | 40.83% | 10.00 pp | -104 | 45 | -2.31 |
| BTC Daily | rf | RandomForest | 694 | 297 | 397 | 42.80% | 40.00% | 43.33% | 7.20 pp | -100 | 42 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 520 | 206 | 314 | 39.62% | 37.08% | 39.17% | 10.38 pp | -108 | 45 | -2.40 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 872 | 371 | 501 | 42.55% | 38.75% | 41.67% | 7.45 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 872 | 369 | 503 | 42.32% | 41.25% | 43.12% | 7.68 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 704 | 278 | 426 | 39.49% | 35.42% | 39.17% | 10.51 pp | -148 | 42 | -3.52 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 694 | 338 | 356 | 48.70% | 45.42% | 48.96% | 1.30 pp | -18 | 42 | -0.43 |
| BTC Daily | transformer | Transformer | 694 | 333 | 361 | 47.98% | 45.83% | 49.38% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 694 | 322 | 372 | 46.40% | 42.50% | 48.33% | 3.60 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 694 | 302 | 392 | 43.52% | 38.33% | 42.50% | 6.48 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 694 | 297 | 397 | 42.80% | 40.00% | 43.33% | 7.20 pp | -100 | 42 | -2.38 |
| BTC Daily | xgb | XGBoost | 704 | 278 | 426 | 39.49% | 35.42% | 39.17% | 10.51 pp | -148 | 42 | -3.52 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 466 | 226 | 240 | 48.50% | 44.17% | 48.50% | 1.50 pp | -14 | 45 | -0.31 |
| BTC Market Hours | nn | NN | 466 | 218 | 248 | 46.78% | 47.50% | 46.78% | 3.22 pp | -30 | 45 | -0.67 |
| BTC Market Hours | transformer | Transformer | 466 | 217 | 249 | 46.57% | 40.83% | 46.57% | 3.43 pp | -32 | 45 | -0.71 |
| BTC Market Hours | rf | RandomForest | 466 | 200 | 266 | 42.92% | 42.92% | 42.92% | 7.08 pp | -66 | 45 | -1.47 |
| BTC Market Hours | lstm | LSTM | 466 | 199 | 267 | 42.70% | 40.42% | 42.70% | 7.30 pp | -68 | 45 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 466 | 189 | 277 | 40.56% | 39.58% | 40.56% | 9.44 pp | -88 | 45 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 520 | 238 | 282 | 45.77% | 46.25% | 46.25% | 4.23 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 520 | 237 | 283 | 45.58% | 42.50% | 46.04% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 520 | 237 | 283 | 45.58% | 47.08% | 46.25% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 520 | 215 | 305 | 41.35% | 41.25% | 41.25% | 8.65 pp | -90 | 45 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 520 | 208 | 312 | 40.00% | 37.92% | 40.83% | 10.00 pp | -104 | 45 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 520 | 206 | 314 | 39.62% | 37.08% | 39.17% | 10.38 pp | -108 | 45 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
