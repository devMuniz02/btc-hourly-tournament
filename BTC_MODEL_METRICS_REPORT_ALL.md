# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T00:09:15.640025+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 154 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 154 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 39 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 39 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 508 | 246 | 262 | 48.43% | 45.00% | 48.33% | 1.57 pp | -16 | 48 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| BTC Market Hours | transformer | Transformer | 508 | 243 | 265 | 47.83% | 46.25% | 48.12% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 736 | 356 | 380 | 48.37% | 47.08% | 48.12% | 1.63 pp | -24 | 43 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 562 | 267 | 295 | 47.51% | 50.83% | 48.75% | 2.49 pp | -28 | 48 | -0.58 |
| BTC Market Hours | nn | NN | 508 | 239 | 269 | 47.05% | 49.17% | 47.92% | 2.95 pp | -30 | 48 | -0.62 |
| BTC Daily | transformer | Transformer | 736 | 351 | 385 | 47.69% | 46.25% | 49.79% | 2.31 pp | -34 | 43 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 913 | 437 | 476 | 47.86% | 50.42% | 47.71% | 2.14 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | nn | NN | 562 | 261 | 301 | 46.44% | 45.83% | 47.92% | 3.56 pp | -40 | 48 | -0.83 |
| Consolidated Hourly | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 562 | 259 | 303 | 46.09% | 48.75% | 46.67% | 3.91 pp | -44 | 48 | -0.92 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 913 | 431 | 482 | 47.21% | 47.92% | 46.46% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 736 | 342 | 394 | 46.47% | 45.00% | 47.29% | 3.53 pp | -52 | 43 | -1.21 |
| BTC Market Hours | lstm | LSTM | 508 | 221 | 287 | 43.50% | 42.92% | 43.54% | 6.50 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 508 | 219 | 289 | 43.11% | 43.75% | 43.33% | 6.89 pp | -70 | 48 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 508 | 208 | 300 | 40.94% | 42.08% | 41.46% | 9.06 pp | -92 | 48 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 562 | 233 | 329 | 41.46% | 42.08% | 40.42% | 8.54 pp | -96 | 48 | -2.00 |
| BTC Hourly | nn | NN | 913 | 405 | 508 | 44.36% | 43.33% | 42.08% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | rf | RandomForest | 913 | 405 | 508 | 44.36% | 43.75% | 43.96% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 562 | 228 | 334 | 40.57% | 39.17% | 40.83% | 9.43 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 562 | 224 | 338 | 39.86% | 40.00% | 38.75% | 10.14 pp | -114 | 48 | -2.38 |
| BTC Daily | lstm | LSTM | 736 | 315 | 421 | 42.80% | 36.25% | 41.04% | 7.20 pp | -106 | 43 | -2.47 |
| Consolidated Hourly | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| BTC Daily | rf | RandomForest | 736 | 312 | 424 | 42.39% | 40.00% | 42.92% | 7.61 pp | -112 | 43 | -2.60 |
| BTC Hourly | lstm | LSTM | 913 | 390 | 523 | 42.72% | 39.58% | 41.25% | 7.28 pp | -133 | 48 | -2.77 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 913 | 382 | 531 | 41.84% | 40.00% | 40.42% | 8.16 pp | -149 | 48 | -3.10 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
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
| Consolidated Hourly | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
