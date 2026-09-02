# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T16:52:06.623036+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1198 | 910 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1073 | 708 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 15:00:00+00:00 | 719 | 470 | 248 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 15:00:00+00:00 | 721 | 524 | 195 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 121 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 18:00:00+00:00 | 121 | 21 | 100 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 470 | 228 | 242 | 48.51% | 44.17% | 48.51% | 1.49 pp | -14 | 46 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 698 | 342 | 356 | 49.00% | 47.08% | 49.38% | 1.00 pp | -14 | 42 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 470 | 221 | 249 | 47.02% | 47.50% | 47.02% | 2.98 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 698 | 335 | 363 | 47.99% | 46.67% | 49.38% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Market Hours | transformer | Transformer | 470 | 219 | 251 | 46.60% | 41.25% | 46.60% | 3.40 pp | -32 | 46 | -0.70 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 876 | 416 | 460 | 47.49% | 48.75% | 47.92% | 2.51 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 524 | 240 | 284 | 45.80% | 46.67% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 524 | 240 | 284 | 45.80% | 47.08% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 524 | 239 | 285 | 45.61% | 42.50% | 46.25% | 4.39 pp | -46 | 45 | -1.02 |
| BTC Hourly | transformer | Transformer | 876 | 414 | 462 | 47.26% | 48.33% | 47.71% | 2.74 pp | -48 | 46 | -1.04 |
| BTC Daily | nn | NN | 698 | 324 | 374 | 46.42% | 42.92% | 48.54% | 3.58 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 470 | 202 | 268 | 42.98% | 42.92% | 42.98% | 7.02 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 470 | 201 | 269 | 42.77% | 40.83% | 42.77% | 7.23 pp | -68 | 46 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 876 | 394 | 482 | 44.98% | 46.67% | 43.96% | 5.02 pp | -88 | 46 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 470 | 191 | 279 | 40.64% | 40.00% | 40.64% | 9.36 pp | -88 | 46 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 524 | 216 | 308 | 41.22% | 40.83% | 41.25% | 8.78 pp | -92 | 45 | -2.04 |
| BTC Hourly | rf | RandomForest | 876 | 390 | 486 | 44.52% | 44.58% | 44.17% | 5.48 pp | -96 | 46 | -2.09 |
| BTC Daily | lstm | LSTM | 698 | 304 | 394 | 43.55% | 38.75% | 42.29% | 6.45 pp | -90 | 42 | -2.14 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Daily | rf | RandomForest | 698 | 300 | 398 | 42.98% | 41.25% | 43.33% | 7.02 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 524 | 209 | 315 | 39.89% | 37.08% | 40.83% | 10.11 pp | -106 | 45 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 524 | 207 | 317 | 39.50% | 37.50% | 38.96% | 10.50 pp | -110 | 45 | -2.44 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 876 | 374 | 502 | 42.69% | 38.75% | 41.88% | 7.31 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 876 | 372 | 504 | 42.47% | 42.08% | 43.33% | 7.53 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 708 | 281 | 427 | 39.69% | 36.25% | 39.58% | 10.31 pp | -146 | 42 | -3.48 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 876 | 416 | 460 | 47.49% | 48.75% | 47.92% | 2.51 pp | -44 | 46 | -0.96 |
| BTC Hourly | transformer | Transformer | 876 | 414 | 462 | 47.26% | 48.33% | 47.71% | 2.74 pp | -48 | 46 | -1.04 |
| BTC Hourly | nn | NN | 876 | 394 | 482 | 44.98% | 46.67% | 43.96% | 5.02 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 876 | 390 | 486 | 44.52% | 44.58% | 44.17% | 5.48 pp | -96 | 46 | -2.09 |
| BTC Hourly | lstm | LSTM | 876 | 374 | 502 | 42.69% | 38.75% | 41.88% | 7.31 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 876 | 372 | 504 | 42.47% | 42.08% | 43.33% | 7.53 pp | -132 | 46 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 698 | 342 | 356 | 49.00% | 47.08% | 49.38% | 1.00 pp | -14 | 42 | -0.33 |
| BTC Daily | transformer | Transformer | 698 | 335 | 363 | 47.99% | 46.67% | 49.38% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 698 | 324 | 374 | 46.42% | 42.92% | 48.54% | 3.58 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 698 | 304 | 394 | 43.55% | 38.75% | 42.29% | 6.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 698 | 300 | 398 | 42.98% | 41.25% | 43.33% | 7.02 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 708 | 281 | 427 | 39.69% | 36.25% | 39.58% | 10.31 pp | -146 | 42 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 470 | 228 | 242 | 48.51% | 44.17% | 48.51% | 1.49 pp | -14 | 46 | -0.30 |
| BTC Market Hours | nn | NN | 470 | 221 | 249 | 47.02% | 47.50% | 47.02% | 2.98 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 470 | 219 | 251 | 46.60% | 41.25% | 46.60% | 3.40 pp | -32 | 46 | -0.70 |
| BTC Market Hours | rf | RandomForest | 470 | 202 | 268 | 42.98% | 42.92% | 42.98% | 7.02 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 470 | 201 | 269 | 42.77% | 40.83% | 42.77% | 7.23 pp | -68 | 46 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 470 | 191 | 279 | 40.64% | 40.00% | 40.64% | 9.36 pp | -88 | 46 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 524 | 240 | 284 | 45.80% | 46.67% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 524 | 240 | 284 | 45.80% | 47.08% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 524 | 239 | 285 | 45.61% | 42.50% | 46.25% | 4.39 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 524 | 216 | 308 | 41.22% | 40.83% | 41.25% | 8.78 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 524 | 209 | 315 | 39.89% | 37.08% | 40.83% | 10.11 pp | -106 | 45 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 524 | 207 | 317 | 39.50% | 37.50% | 38.96% | 10.50 pp | -110 | 45 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 63 | 58 | 52.07% | 52.07% | 52.07% | 2.07 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 57 | 64 | 47.11% | 47.11% | 47.11% | 2.89 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
