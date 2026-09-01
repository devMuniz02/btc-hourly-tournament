# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T21:32:31.330461+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1185 | 897 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1060 | 695 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 20:00:00+00:00 | 698 | 457 | 240 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 20:00:00+00:00 | 700 | 511 | 187 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 457 | 221 | 236 | 48.36% | 44.17% | 48.36% | 1.64 pp | -15 | 45 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 685 | 334 | 351 | 48.76% | 46.25% | 49.38% | 1.24 pp | -17 | 41 | -0.41 |
| BTC Market Hours | nn | NN | 457 | 215 | 242 | 47.05% | 48.75% | 47.05% | 2.95 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 685 | 329 | 356 | 48.03% | 45.83% | 49.38% | 1.97 pp | -27 | 41 | -0.66 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 457 | 210 | 247 | 45.95% | 39.58% | 45.95% | 4.05 pp | -37 | 45 | -0.82 |
| BTC Market Hours Daily | nn | NN | 511 | 234 | 277 | 45.79% | 43.33% | 46.67% | 4.21 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 511 | 233 | 278 | 45.60% | 45.83% | 46.04% | 4.40 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 863 | 407 | 456 | 47.16% | 45.83% | 47.08% | 2.84 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 685 | 320 | 365 | 46.72% | 42.92% | 48.96% | 3.28 pp | -45 | 41 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | transformer | Transformer | 511 | 231 | 280 | 45.21% | 46.25% | 45.62% | 4.79 pp | -49 | 44 | -1.11 |
| BTC Hourly | transformer | Transformer | 863 | 405 | 458 | 46.93% | 47.08% | 46.67% | 3.07 pp | -53 | 46 | -1.15 |
| BTC Market Hours | rf | RandomForest | 457 | 198 | 259 | 43.33% | 43.33% | 43.33% | 6.67 pp | -61 | 45 | -1.36 |
| BTC Market Hours | lstm | LSTM | 457 | 194 | 263 | 42.45% | 40.00% | 42.45% | 7.55 pp | -69 | 45 | -1.53 |
| BTC Hourly | nn | NN | 863 | 389 | 474 | 45.08% | 45.83% | 44.17% | 4.92 pp | -85 | 46 | -1.85 |
| BTC Market Hours | xgb | XGBoost | 457 | 186 | 271 | 40.70% | 39.17% | 40.70% | 9.30 pp | -85 | 45 | -1.89 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 511 | 212 | 299 | 41.49% | 41.67% | 41.67% | 8.51 pp | -87 | 44 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 863 | 383 | 480 | 44.38% | 43.33% | 43.75% | 5.62 pp | -97 | 46 | -2.11 |
| BTC Daily | lstm | LSTM | 685 | 299 | 386 | 43.65% | 38.75% | 42.71% | 6.35 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 511 | 205 | 306 | 40.12% | 38.33% | 40.83% | 9.88 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 685 | 294 | 391 | 42.92% | 40.83% | 43.33% | 7.08 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 511 | 202 | 309 | 39.53% | 36.25% | 39.17% | 10.47 pp | -107 | 44 | -2.43 |
| BTC Hourly | lstm | LSTM | 863 | 367 | 496 | 42.53% | 37.92% | 41.88% | 7.47 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 863 | 364 | 499 | 42.18% | 40.42% | 42.92% | 7.82 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 695 | 275 | 420 | 39.57% | 35.00% | 39.38% | 10.43 pp | -145 | 41 | -3.54 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 863 | 407 | 456 | 47.16% | 45.83% | 47.08% | 2.84 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 863 | 405 | 458 | 46.93% | 47.08% | 46.67% | 3.07 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 863 | 389 | 474 | 45.08% | 45.83% | 44.17% | 4.92 pp | -85 | 46 | -1.85 |
| BTC Hourly | rf | RandomForest | 863 | 383 | 480 | 44.38% | 43.33% | 43.75% | 5.62 pp | -97 | 46 | -2.11 |
| BTC Hourly | lstm | LSTM | 863 | 367 | 496 | 42.53% | 37.92% | 41.88% | 7.47 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 863 | 364 | 499 | 42.18% | 40.42% | 42.92% | 7.82 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 685 | 334 | 351 | 48.76% | 46.25% | 49.38% | 1.24 pp | -17 | 41 | -0.41 |
| BTC Daily | transformer | Transformer | 685 | 329 | 356 | 48.03% | 45.83% | 49.38% | 1.97 pp | -27 | 41 | -0.66 |
| BTC Daily | nn | NN | 685 | 320 | 365 | 46.72% | 42.92% | 48.96% | 3.28 pp | -45 | 41 | -1.10 |
| BTC Daily | lstm | LSTM | 685 | 299 | 386 | 43.65% | 38.75% | 42.71% | 6.35 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 685 | 294 | 391 | 42.92% | 40.83% | 43.33% | 7.08 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 695 | 275 | 420 | 39.57% | 35.00% | 39.38% | 10.43 pp | -145 | 41 | -3.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 457 | 221 | 236 | 48.36% | 44.17% | 48.36% | 1.64 pp | -15 | 45 | -0.33 |
| BTC Market Hours | nn | NN | 457 | 215 | 242 | 47.05% | 48.75% | 47.05% | 2.95 pp | -27 | 45 | -0.60 |
| BTC Market Hours | transformer | Transformer | 457 | 210 | 247 | 45.95% | 39.58% | 45.95% | 4.05 pp | -37 | 45 | -0.82 |
| BTC Market Hours | rf | RandomForest | 457 | 198 | 259 | 43.33% | 43.33% | 43.33% | 6.67 pp | -61 | 45 | -1.36 |
| BTC Market Hours | lstm | LSTM | 457 | 194 | 263 | 42.45% | 40.00% | 42.45% | 7.55 pp | -69 | 45 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 457 | 186 | 271 | 40.70% | 39.17% | 40.70% | 9.30 pp | -85 | 45 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 511 | 234 | 277 | 45.79% | 43.33% | 46.67% | 4.21 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 511 | 233 | 278 | 45.60% | 45.83% | 46.04% | 4.40 pp | -45 | 44 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 511 | 231 | 280 | 45.21% | 46.25% | 45.62% | 4.79 pp | -49 | 44 | -1.11 |
| BTC Market Hours Daily | rf | RandomForest | 511 | 212 | 299 | 41.49% | 41.67% | 41.67% | 8.51 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 511 | 205 | 306 | 40.12% | 38.33% | 40.83% | 9.88 pp | -101 | 44 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 511 | 202 | 309 | 39.53% | 36.25% | 39.17% | 10.47 pp | -107 | 44 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
