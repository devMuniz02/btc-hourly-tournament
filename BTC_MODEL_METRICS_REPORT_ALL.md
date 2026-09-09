# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T04:13:20.521737+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1303 | 1015 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1179 | 814 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 912 | 576 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 913 | 629 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 576 | 280 | 296 | 48.61% | 47.08% | 47.71% | 1.39 pp | -16 | 54 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 576 | 275 | 301 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 576 | 272 | 304 | 47.22% | 47.50% | 47.08% | 2.78 pp | -32 | 54 | -0.59 |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Market Hours Daily | nn | NN | 629 | 294 | 335 | 46.74% | 47.50% | 47.92% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 629 | 294 | 335 | 46.74% | 49.17% | 47.29% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 629 | 292 | 337 | 46.42% | 47.92% | 46.67% | 3.58 pp | -45 | 54 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 981 | 469 | 512 | 47.81% | 51.25% | 46.88% | 2.19 pp | -43 | 51 | -0.84 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 373 | 431 | 46.39% | 39.58% | 46.04% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 981 | 454 | 527 | 46.28% | 43.33% | 43.33% | 3.72 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 576 | 249 | 327 | 43.23% | 42.50% | 43.33% | 6.77 pp | -78 | 54 | -1.44 |
| BTC Market Hours | rf | RandomForest | 576 | 248 | 328 | 43.06% | 45.00% | 43.33% | 6.94 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 576 | 248 | 328 | 43.06% | 46.25% | 43.12% | 6.94 pp | -80 | 54 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 629 | 263 | 366 | 41.81% | 43.75% | 40.62% | 8.19 pp | -103 | 54 | -1.91 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 629 | 261 | 368 | 41.49% | 45.00% | 41.04% | 8.51 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 629 | 257 | 372 | 40.86% | 42.08% | 40.21% | 9.14 pp | -115 | 54 | -2.13 |
| BTC Hourly | rf | RandomForest | 981 | 434 | 547 | 44.24% | 42.08% | 42.71% | 5.76 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 981 | 433 | 548 | 44.14% | 41.67% | 42.29% | 5.86 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| BTC Daily | lstm | LSTM | 804 | 339 | 465 | 42.16% | 35.42% | 40.42% | 7.84 pp | -126 | 46 | -2.74 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 981 | 417 | 564 | 42.51% | 37.50% | 40.83% | 7.49 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 804 | 334 | 470 | 41.54% | 37.08% | 41.04% | 8.46 pp | -136 | 46 | -2.96 |
| BTC Hourly | xgb | XGBoost | 981 | 405 | 576 | 41.28% | 36.25% | 38.96% | 8.72 pp | -171 | 51 | -3.35 |
| BTC Daily | xgb | XGBoost | 814 | 316 | 498 | 38.82% | 35.00% | 35.42% | 11.18 pp | -182 | 46 | -3.96 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 981 | 469 | 512 | 47.81% | 51.25% | 46.88% | 2.19 pp | -43 | 51 | -0.84 |
| BTC Hourly | transformer | Transformer | 981 | 454 | 527 | 46.28% | 43.33% | 43.33% | 3.72 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 981 | 434 | 547 | 44.24% | 42.08% | 42.71% | 5.76 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 981 | 433 | 548 | 44.14% | 41.67% | 42.29% | 5.86 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 981 | 417 | 564 | 42.51% | 37.50% | 40.83% | 7.49 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 981 | 405 | 576 | 41.28% | 36.25% | 38.96% | 8.72 pp | -171 | 51 | -3.35 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 373 | 431 | 46.39% | 39.58% | 46.04% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 804 | 339 | 465 | 42.16% | 35.42% | 40.42% | 7.84 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 804 | 334 | 470 | 41.54% | 37.08% | 41.04% | 8.46 pp | -136 | 46 | -2.96 |
| BTC Daily | xgb | XGBoost | 814 | 316 | 498 | 38.82% | 35.00% | 35.42% | 11.18 pp | -182 | 46 | -3.96 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 576 | 280 | 296 | 48.61% | 47.08% | 47.71% | 1.39 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 576 | 275 | 301 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 576 | 272 | 304 | 47.22% | 47.50% | 47.08% | 2.78 pp | -32 | 54 | -0.59 |
| BTC Market Hours | lstm | LSTM | 576 | 249 | 327 | 43.23% | 42.50% | 43.33% | 6.77 pp | -78 | 54 | -1.44 |
| BTC Market Hours | rf | RandomForest | 576 | 248 | 328 | 43.06% | 45.00% | 43.33% | 6.94 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 576 | 248 | 328 | 43.06% | 46.25% | 43.12% | 6.94 pp | -80 | 54 | -1.48 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 629 | 294 | 335 | 46.74% | 47.50% | 47.92% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 629 | 294 | 335 | 46.74% | 49.17% | 47.29% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 629 | 292 | 337 | 46.42% | 47.92% | 46.67% | 3.58 pp | -45 | 54 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 629 | 263 | 366 | 41.81% | 43.75% | 40.62% | 8.19 pp | -103 | 54 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 629 | 261 | 368 | 41.49% | 45.00% | 41.04% | 8.51 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 629 | 257 | 372 | 40.86% | 42.08% | 40.21% | 9.14 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
