# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T00:56:37.799572+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1235 | 947 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1111 | 746 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 23:00:00+00:00 | 791 | 508 | 282 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 23:00:00+00:00 | 793 | 562 | 229 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 156 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 508 | 246 | 262 | 48.43% | 45.00% | 48.33% | 1.57 pp | -16 | 48 | -0.33 |
| BTC Market Hours | transformer | Transformer | 508 | 243 | 265 | 47.83% | 46.25% | 48.12% | 2.17 pp | -22 | 48 | -0.46 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 736 | 356 | 380 | 48.37% | 47.08% | 48.12% | 1.63 pp | -24 | 43 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 562 | 267 | 295 | 47.51% | 50.83% | 48.75% | 2.49 pp | -28 | 48 | -0.58 |
| BTC Market Hours | nn | NN | 508 | 239 | 269 | 47.05% | 49.17% | 47.92% | 2.95 pp | -30 | 48 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| BTC Daily | transformer | Transformer | 736 | 351 | 385 | 47.69% | 46.25% | 49.79% | 2.31 pp | -34 | 43 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 913 | 437 | 476 | 47.86% | 50.42% | 47.71% | 2.14 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | nn | NN | 562 | 261 | 301 | 46.44% | 45.83% | 47.92% | 3.56 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 562 | 259 | 303 | 46.09% | 48.75% | 46.67% | 3.91 pp | -44 | 48 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 913 | 431 | 482 | 47.21% | 47.92% | 46.46% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 736 | 342 | 394 | 46.47% | 45.00% | 47.29% | 3.53 pp | -52 | 43 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 508 | 221 | 287 | 43.50% | 42.92% | 43.54% | 6.50 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 508 | 219 | 289 | 43.11% | 43.75% | 43.33% | 6.89 pp | -70 | 48 | -1.46 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| BTC Market Hours | xgb | XGBoost | 508 | 208 | 300 | 40.94% | 42.08% | 41.46% | 9.06 pp | -92 | 48 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 562 | 233 | 329 | 41.46% | 42.08% | 40.42% | 8.54 pp | -96 | 48 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | nn | NN | 913 | 405 | 508 | 44.36% | 43.33% | 42.08% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | rf | RandomForest | 913 | 405 | 508 | 44.36% | 43.75% | 43.96% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 562 | 228 | 334 | 40.57% | 39.17% | 40.83% | 9.43 pp | -106 | 48 | -2.21 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 562 | 224 | 338 | 39.86% | 40.00% | 38.75% | 10.14 pp | -114 | 48 | -2.38 |
| BTC Daily | lstm | LSTM | 736 | 315 | 421 | 42.80% | 36.25% | 41.04% | 7.20 pp | -106 | 43 | -2.47 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 736 | 312 | 424 | 42.39% | 40.00% | 42.92% | 7.61 pp | -112 | 43 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 913 | 390 | 523 | 42.72% | 39.58% | 41.25% | 7.28 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 913 | 382 | 531 | 41.84% | 40.00% | 40.42% | 8.16 pp | -149 | 48 | -3.10 |
| BTC Daily | xgb | XGBoost | 746 | 294 | 452 | 39.41% | 36.25% | 37.71% | 10.59 pp | -158 | 43 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 913 | 437 | 476 | 47.86% | 50.42% | 47.71% | 2.14 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 913 | 431 | 482 | 47.21% | 47.92% | 46.46% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 913 | 405 | 508 | 44.36% | 43.33% | 42.08% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | rf | RandomForest | 913 | 405 | 508 | 44.36% | 43.75% | 43.96% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 913 | 390 | 523 | 42.72% | 39.58% | 41.25% | 7.28 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 913 | 382 | 531 | 41.84% | 40.00% | 40.42% | 8.16 pp | -149 | 48 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 736 | 356 | 380 | 48.37% | 47.08% | 48.12% | 1.63 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 736 | 351 | 385 | 47.69% | 46.25% | 49.79% | 2.31 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 736 | 342 | 394 | 46.47% | 45.00% | 47.29% | 3.53 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 736 | 315 | 421 | 42.80% | 36.25% | 41.04% | 7.20 pp | -106 | 43 | -2.47 |
| BTC Daily | rf | RandomForest | 736 | 312 | 424 | 42.39% | 40.00% | 42.92% | 7.61 pp | -112 | 43 | -2.60 |
| BTC Daily | xgb | XGBoost | 746 | 294 | 452 | 39.41% | 36.25% | 37.71% | 10.59 pp | -158 | 43 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 508 | 246 | 262 | 48.43% | 45.00% | 48.33% | 1.57 pp | -16 | 48 | -0.33 |
| BTC Market Hours | transformer | Transformer | 508 | 243 | 265 | 47.83% | 46.25% | 48.12% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Market Hours | nn | NN | 508 | 239 | 269 | 47.05% | 49.17% | 47.92% | 2.95 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 508 | 221 | 287 | 43.50% | 42.92% | 43.54% | 6.50 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 508 | 219 | 289 | 43.11% | 43.75% | 43.33% | 6.89 pp | -70 | 48 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 508 | 208 | 300 | 40.94% | 42.08% | 41.46% | 9.06 pp | -92 | 48 | -1.92 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 562 | 267 | 295 | 47.51% | 50.83% | 48.75% | 2.49 pp | -28 | 48 | -0.58 |
| BTC Market Hours Daily | nn | NN | 562 | 261 | 301 | 46.44% | 45.83% | 47.92% | 3.56 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 562 | 259 | 303 | 46.09% | 48.75% | 46.67% | 3.91 pp | -44 | 48 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 562 | 233 | 329 | 41.46% | 42.08% | 40.42% | 8.54 pp | -96 | 48 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 562 | 228 | 334 | 40.57% | 39.17% | 40.83% | 9.43 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 562 | 224 | 338 | 39.86% | 40.00% | 38.75% | 10.14 pp | -114 | 48 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
