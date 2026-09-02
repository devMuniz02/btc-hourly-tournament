# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T09:33:18.224411+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1193 | 905 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1069 | 704 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 711 | 466 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 713 | 520 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 116 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T15:00:00+00:00 | 117 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 466 | 226 | 240 | 48.50% | 44.17% | 48.50% | 1.50 pp | -14 | 45 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 694 | 338 | 356 | 48.70% | 45.42% | 48.96% | 1.30 pp | -18 | 42 | -0.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| BTC Daily | transformer | Transformer | 694 | 333 | 361 | 47.98% | 45.83% | 49.38% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Market Hours | nn | NN | 466 | 218 | 248 | 46.78% | 47.50% | 46.78% | 3.22 pp | -30 | 45 | -0.67 |
| BTC Market Hours | transformer | Transformer | 466 | 217 | 249 | 46.57% | 40.83% | 46.57% | 3.43 pp | -32 | 45 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 520 | 238 | 282 | 45.77% | 46.25% | 46.25% | 4.23 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 871 | 412 | 459 | 47.30% | 47.50% | 47.50% | 2.70 pp | -47 | 46 | -1.02 |
| BTC Market Hours Daily | nn | NN | 520 | 237 | 283 | 45.58% | 42.50% | 46.04% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 520 | 237 | 283 | 45.58% | 47.08% | 46.25% | 4.42 pp | -46 | 45 | -1.02 |
| BTC Hourly | transformer | Transformer | 871 | 410 | 461 | 47.07% | 47.92% | 47.08% | 2.93 pp | -51 | 46 | -1.11 |
| BTC Daily | nn | NN | 694 | 322 | 372 | 46.40% | 42.50% | 48.33% | 3.60 pp | -50 | 42 | -1.19 |
| BTC Market Hours | rf | RandomForest | 466 | 200 | 266 | 42.92% | 42.92% | 42.92% | 7.08 pp | -66 | 45 | -1.47 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 466 | 199 | 267 | 42.70% | 40.42% | 42.70% | 7.30 pp | -68 | 45 | -1.51 |
| Consolidated Hourly | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |
| BTC Hourly | nn | NN | 871 | 392 | 479 | 45.01% | 46.25% | 43.96% | 4.99 pp | -87 | 46 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 466 | 189 | 277 | 40.56% | 39.58% | 40.56% | 9.44 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 520 | 215 | 305 | 41.35% | 41.25% | 41.25% | 8.65 pp | -90 | 45 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 871 | 389 | 482 | 44.66% | 45.42% | 44.17% | 5.34 pp | -93 | 46 | -2.02 |
| BTC Daily | lstm | LSTM | 694 | 301 | 393 | 43.37% | 37.92% | 42.29% | 6.63 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 520 | 208 | 312 | 40.00% | 37.92% | 40.83% | 10.00 pp | -104 | 45 | -2.31 |
| BTC Daily | rf | RandomForest | 694 | 297 | 397 | 42.80% | 40.00% | 43.33% | 7.20 pp | -100 | 42 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 520 | 206 | 314 | 39.62% | 37.08% | 39.17% | 10.38 pp | -108 | 45 | -2.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 871 | 370 | 501 | 42.48% | 38.33% | 41.67% | 7.52 pp | -131 | 46 | -2.85 |
| BTC Hourly | xgb | XGBoost | 871 | 368 | 503 | 42.25% | 40.83% | 43.12% | 7.75 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 704 | 278 | 426 | 39.49% | 35.42% | 39.17% | 10.51 pp | -148 | 42 | -3.52 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 871 | 412 | 459 | 47.30% | 47.50% | 47.50% | 2.70 pp | -47 | 46 | -1.02 |
| BTC Hourly | transformer | Transformer | 871 | 410 | 461 | 47.07% | 47.92% | 47.08% | 2.93 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 871 | 392 | 479 | 45.01% | 46.25% | 43.96% | 4.99 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 871 | 389 | 482 | 44.66% | 45.42% | 44.17% | 5.34 pp | -93 | 46 | -2.02 |
| BTC Hourly | lstm | LSTM | 871 | 370 | 501 | 42.48% | 38.33% | 41.67% | 7.52 pp | -131 | 46 | -2.85 |
| BTC Hourly | xgb | XGBoost | 871 | 368 | 503 | 42.25% | 40.83% | 43.12% | 7.75 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 694 | 338 | 356 | 48.70% | 45.42% | 48.96% | 1.30 pp | -18 | 42 | -0.43 |
| BTC Daily | transformer | Transformer | 694 | 333 | 361 | 47.98% | 45.83% | 49.38% | 2.02 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 694 | 322 | 372 | 46.40% | 42.50% | 48.33% | 3.60 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 694 | 301 | 393 | 43.37% | 37.92% | 42.29% | 6.63 pp | -92 | 42 | -2.19 |
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
| Consolidated Hourly | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 116 | 50 | 66 | 43.10% | 43.10% | 43.10% | 6.90 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
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
