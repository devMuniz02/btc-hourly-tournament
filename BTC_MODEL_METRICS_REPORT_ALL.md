# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T16:22:43.161595+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1295 | 1007 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1171 | 806 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 15:00:00+00:00 | 895 | 568 | 326 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 15:00:00+00:00 | 897 | 622 | 273 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 210 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 210 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 210 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 211 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 568 | 277 | 291 | 48.77% | 47.50% | 48.12% | 1.23 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 568 | 270 | 298 | 47.54% | 51.67% | 49.17% | 2.46 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 796 | 385 | 411 | 48.37% | 47.08% | 48.12% | 1.63 pp | -26 | 46 | -0.57 |
| Consolidated Hourly | rf | RandomForest | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 14 | -0.57 |
| BTC Market Hours | transformer | Transformer | 568 | 267 | 301 | 47.01% | 47.08% | 47.08% | 2.99 pp | -34 | 53 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 14 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 622 | 291 | 331 | 46.78% | 49.17% | 47.71% | 3.22 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 622 | 290 | 332 | 46.62% | 47.50% | 47.92% | 3.38 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 622 | 289 | 333 | 46.46% | 48.75% | 47.08% | 3.54 pp | -44 | 53 | -0.83 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 973 | 462 | 511 | 47.48% | 49.17% | 46.46% | 2.52 pp | -49 | 51 | -0.96 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| BTC Daily | transformer | Transformer | 796 | 370 | 426 | 46.48% | 40.00% | 46.46% | 3.52 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 796 | 369 | 427 | 46.36% | 44.58% | 45.00% | 3.64 pp | -58 | 46 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| BTC Hourly | transformer | Transformer | 973 | 453 | 520 | 46.56% | 44.58% | 43.75% | 3.44 pp | -67 | 51 | -1.31 |
| BTC Market Hours | lstm | LSTM | 568 | 247 | 321 | 43.49% | 42.92% | 43.96% | 6.51 pp | -74 | 53 | -1.40 |
| BTC Market Hours | rf | RandomForest | 568 | 245 | 323 | 43.13% | 45.83% | 43.75% | 6.87 pp | -78 | 53 | -1.47 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 568 | 244 | 324 | 42.96% | 46.67% | 43.54% | 7.04 pp | -80 | 53 | -1.51 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 14 | -1.86 |
| Consolidated Daily/Hourly Refresh | nn | NN | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 622 | 260 | 362 | 41.80% | 43.75% | 40.83% | 8.20 pp | -102 | 53 | -1.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 622 | 257 | 365 | 41.32% | 44.17% | 40.83% | 8.68 pp | -108 | 53 | -2.04 |
| BTC Hourly | rf | RandomForest | 973 | 431 | 542 | 44.30% | 42.08% | 42.71% | 5.70 pp | -111 | 51 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 622 | 253 | 369 | 40.68% | 40.83% | 39.79% | 9.32 pp | -116 | 53 | -2.19 |
| BTC Hourly | nn | NN | 973 | 430 | 543 | 44.19% | 41.25% | 42.50% | 5.81 pp | -113 | 51 | -2.22 |
| Consolidated Hourly | transformer | Transformer | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 14 | -2.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 796 | 334 | 462 | 41.96% | 33.75% | 40.00% | 8.04 pp | -128 | 46 | -2.78 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 973 | 414 | 559 | 42.55% | 37.92% | 41.25% | 7.45 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 796 | 332 | 464 | 41.71% | 37.50% | 41.25% | 8.29 pp | -132 | 46 | -2.87 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 973 | 401 | 572 | 41.21% | 35.83% | 38.75% | 8.79 pp | -171 | 51 | -3.35 |
| BTC Daily | xgb | XGBoost | 806 | 314 | 492 | 38.96% | 35.42% | 35.83% | 11.04 pp | -178 | 46 | -3.87 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 973 | 462 | 511 | 47.48% | 49.17% | 46.46% | 2.52 pp | -49 | 51 | -0.96 |
| BTC Hourly | transformer | Transformer | 973 | 453 | 520 | 46.56% | 44.58% | 43.75% | 3.44 pp | -67 | 51 | -1.31 |
| BTC Hourly | rf | RandomForest | 973 | 431 | 542 | 44.30% | 42.08% | 42.71% | 5.70 pp | -111 | 51 | -2.18 |
| BTC Hourly | nn | NN | 973 | 430 | 543 | 44.19% | 41.25% | 42.50% | 5.81 pp | -113 | 51 | -2.22 |
| BTC Hourly | lstm | LSTM | 973 | 414 | 559 | 42.55% | 37.92% | 41.25% | 7.45 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 973 | 401 | 572 | 41.21% | 35.83% | 38.75% | 8.79 pp | -171 | 51 | -3.35 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 796 | 385 | 411 | 48.37% | 47.08% | 48.12% | 1.63 pp | -26 | 46 | -0.57 |
| BTC Daily | transformer | Transformer | 796 | 370 | 426 | 46.48% | 40.00% | 46.46% | 3.52 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 796 | 369 | 427 | 46.36% | 44.58% | 45.00% | 3.64 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 796 | 334 | 462 | 41.96% | 33.75% | 40.00% | 8.04 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 796 | 332 | 464 | 41.71% | 37.50% | 41.25% | 8.29 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 806 | 314 | 492 | 38.96% | 35.42% | 35.83% | 11.04 pp | -178 | 46 | -3.87 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 568 | 277 | 291 | 48.77% | 47.50% | 48.12% | 1.23 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 568 | 270 | 298 | 47.54% | 51.67% | 49.17% | 2.46 pp | -28 | 53 | -0.53 |
| BTC Market Hours | transformer | Transformer | 568 | 267 | 301 | 47.01% | 47.08% | 47.08% | 2.99 pp | -34 | 53 | -0.64 |
| BTC Market Hours | lstm | LSTM | 568 | 247 | 321 | 43.49% | 42.92% | 43.96% | 6.51 pp | -74 | 53 | -1.40 |
| BTC Market Hours | rf | RandomForest | 568 | 245 | 323 | 43.13% | 45.83% | 43.75% | 6.87 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 568 | 244 | 324 | 42.96% | 46.67% | 43.54% | 7.04 pp | -80 | 53 | -1.51 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 622 | 291 | 331 | 46.78% | 49.17% | 47.71% | 3.22 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 622 | 290 | 332 | 46.62% | 47.50% | 47.92% | 3.38 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 622 | 289 | 333 | 46.46% | 48.75% | 47.08% | 3.54 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 622 | 260 | 362 | 41.80% | 43.75% | 40.83% | 8.20 pp | -102 | 53 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 622 | 257 | 365 | 41.32% | 44.17% | 40.83% | 8.68 pp | -108 | 53 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 622 | 253 | 369 | 40.68% | 40.83% | 39.79% | 9.32 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 14 | -0.71 |
| Consolidated Hourly | lstm | LSTM | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | nn | NN | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 14 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 14 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 210 | 96 | 114 | 45.71% | 45.71% | 45.71% | 4.29 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 14 | -1.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 210 | 89 | 121 | 42.38% | 42.38% | 42.38% | 7.62 pp | -32 | 14 | -2.29 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
