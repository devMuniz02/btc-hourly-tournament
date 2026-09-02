# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T16:20:56.983947+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1197 | 909 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1073 | 708 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 15:00:00+00:00 | 719 | 470 | 248 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 15:00:00+00:00 | 721 | 524 | 195 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 121 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 121 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 121 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 122 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 470 | 228 | 242 | 48.51% | 44.17% | 48.51% | 1.49 pp | -14 | 46 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 698 | 342 | 356 | 49.00% | 47.08% | 49.38% | 1.00 pp | -14 | 42 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 470 | 221 | 249 | 47.02% | 47.50% | 47.02% | 2.98 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 698 | 335 | 363 | 47.99% | 46.67% | 49.38% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Market Hours | transformer | Transformer | 470 | 219 | 251 | 46.60% | 41.25% | 46.60% | 3.40 pp | -32 | 46 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 524 | 240 | 284 | 45.80% | 46.67% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 524 | 240 | 284 | 45.80% | 47.08% | 46.46% | 4.20 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 875 | 415 | 460 | 47.43% | 48.75% | 47.71% | 2.57 pp | -45 | 46 | -0.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 875 | 414 | 461 | 47.31% | 48.75% | 47.71% | 2.69 pp | -47 | 46 | -1.02 |
| BTC Market Hours Daily | nn | NN | 524 | 239 | 285 | 45.61% | 42.50% | 46.25% | 4.39 pp | -46 | 45 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 698 | 324 | 374 | 46.42% | 42.92% | 48.54% | 3.58 pp | -50 | 42 | -1.19 |
| BTC Market Hours | rf | RandomForest | 470 | 202 | 268 | 42.98% | 42.92% | 42.98% | 7.02 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 470 | 201 | 269 | 42.77% | 40.83% | 42.77% | 7.23 pp | -68 | 46 | -1.48 |
| Consolidated Hourly | nn | NN | 121 | 53 | 68 | 43.80% | 43.80% | 43.80% | 6.20 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 53 | 68 | 43.80% | 43.80% | 43.80% | 6.20 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |
| BTC Hourly | nn | NN | 875 | 394 | 481 | 45.03% | 46.67% | 43.96% | 4.97 pp | -87 | 46 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 470 | 191 | 279 | 40.64% | 40.00% | 40.64% | 9.36 pp | -88 | 46 | -1.91 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 524 | 216 | 308 | 41.22% | 40.83% | 41.25% | 8.78 pp | -92 | 45 | -2.04 |
| BTC Hourly | rf | RandomForest | 875 | 390 | 485 | 44.57% | 45.00% | 44.17% | 5.43 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 698 | 304 | 394 | 43.55% | 38.75% | 42.29% | 6.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 698 | 301 | 397 | 43.12% | 41.67% | 43.54% | 6.88 pp | -96 | 42 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 524 | 209 | 315 | 39.89% | 37.08% | 40.83% | 10.11 pp | -106 | 45 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 524 | 207 | 317 | 39.50% | 37.50% | 38.96% | 10.50 pp | -110 | 45 | -2.44 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 875 | 373 | 502 | 42.63% | 38.75% | 41.88% | 7.37 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 875 | 371 | 504 | 42.40% | 42.08% | 43.12% | 7.60 pp | -133 | 46 | -2.89 |
| BTC Daily | xgb | XGBoost | 708 | 282 | 426 | 39.83% | 36.67% | 39.79% | 10.17 pp | -144 | 42 | -3.43 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 4 | 18 | 18.18% | 18.18% | 18.18% | 31.82 pp | -14 | 2 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 875 | 415 | 460 | 47.43% | 48.75% | 47.71% | 2.57 pp | -45 | 46 | -0.98 |
| BTC Hourly | transformer | Transformer | 875 | 414 | 461 | 47.31% | 48.75% | 47.71% | 2.69 pp | -47 | 46 | -1.02 |
| BTC Hourly | nn | NN | 875 | 394 | 481 | 45.03% | 46.67% | 43.96% | 4.97 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 875 | 390 | 485 | 44.57% | 45.00% | 44.17% | 5.43 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 875 | 373 | 502 | 42.63% | 38.75% | 41.88% | 7.37 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 875 | 371 | 504 | 42.40% | 42.08% | 43.12% | 7.60 pp | -133 | 46 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 698 | 342 | 356 | 49.00% | 47.08% | 49.38% | 1.00 pp | -14 | 42 | -0.33 |
| BTC Daily | transformer | Transformer | 698 | 335 | 363 | 47.99% | 46.67% | 49.38% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 698 | 324 | 374 | 46.42% | 42.92% | 48.54% | 3.58 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 698 | 304 | 394 | 43.55% | 38.75% | 42.29% | 6.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 698 | 301 | 397 | 43.12% | 41.67% | 43.54% | 6.88 pp | -96 | 42 | -2.29 |
| BTC Daily | xgb | XGBoost | 708 | 282 | 426 | 39.83% | 36.67% | 39.79% | 10.17 pp | -144 | 42 | -3.43 |

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
| Consolidated Hourly | rf | RandomForest | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 121 | 53 | 68 | 43.80% | 43.80% | 43.80% | 6.20 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 121 | 61 | 60 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 121 | 58 | 63 | 47.93% | 47.93% | 47.93% | 2.07 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 121 | 53 | 68 | 43.80% | 43.80% | 43.80% | 6.20 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 4 | 18 | 18.18% | 18.18% | 18.18% | 31.82 pp | -14 | 2 | -7.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
