# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T08:23:59.686714+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1337 | 1049 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1213 | 848 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 972 | 610 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 974 | 664 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 610 | 294 | 316 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 610 | 291 | 319 | 47.70% | 50.42% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours Daily | nn | NN | 664 | 313 | 351 | 47.14% | 50.00% | 48.12% | 2.86 pp | -38 | 56 | -0.68 |
| BTC Market Hours | transformer | Transformer | 610 | 285 | 325 | 46.72% | 46.25% | 45.62% | 3.28 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 664 | 311 | 353 | 46.84% | 49.17% | 47.08% | 3.16 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 664 | 309 | 355 | 46.54% | 48.33% | 47.92% | 3.46 pp | -46 | 56 | -0.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 838 | 399 | 439 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1015 | 480 | 535 | 47.29% | 47.92% | 45.83% | 2.71 pp | -55 | 52 | -1.06 |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 838 | 390 | 448 | 46.54% | 45.00% | 45.21% | 3.46 pp | -58 | 48 | -1.21 |
| BTC Hourly | transformer | Transformer | 1015 | 474 | 541 | 46.70% | 46.25% | 44.58% | 3.30 pp | -67 | 52 | -1.29 |
| BTC Daily | transformer | Transformer | 838 | 386 | 452 | 46.06% | 37.92% | 44.17% | 3.94 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 610 | 261 | 349 | 42.79% | 42.92% | 42.92% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 610 | 261 | 349 | 42.79% | 43.33% | 42.08% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 610 | 259 | 351 | 42.46% | 45.83% | 43.33% | 7.54 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 664 | 276 | 388 | 41.57% | 43.75% | 41.67% | 8.43 pp | -112 | 56 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 664 | 272 | 392 | 40.96% | 44.17% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 664 | 272 | 392 | 40.96% | 43.33% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1015 | 447 | 568 | 44.04% | 41.67% | 40.62% | 5.96 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1015 | 445 | 570 | 43.84% | 40.42% | 42.29% | 6.16 pp | -125 | 52 | -2.40 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 838 | 353 | 485 | 42.12% | 35.83% | 39.79% | 7.88 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| BTC Daily | rf | RandomForest | 838 | 348 | 490 | 41.53% | 37.08% | 40.83% | 8.47 pp | -142 | 48 | -2.96 |
| BTC Hourly | lstm | LSTM | 1015 | 428 | 587 | 42.17% | 35.42% | 39.38% | 7.83 pp | -159 | 52 | -3.06 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1015 | 415 | 600 | 40.89% | 34.58% | 37.71% | 9.11 pp | -185 | 52 | -3.56 |
| BTC Daily | xgb | XGBoost | 848 | 335 | 513 | 39.50% | 37.50% | 36.88% | 10.50 pp | -178 | 48 | -3.71 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1015 | 480 | 535 | 47.29% | 47.92% | 45.83% | 2.71 pp | -55 | 52 | -1.06 |
| BTC Hourly | transformer | Transformer | 1015 | 474 | 541 | 46.70% | 46.25% | 44.58% | 3.30 pp | -67 | 52 | -1.29 |
| BTC Hourly | nn | NN | 1015 | 447 | 568 | 44.04% | 41.67% | 40.62% | 5.96 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1015 | 445 | 570 | 43.84% | 40.42% | 42.29% | 6.16 pp | -125 | 52 | -2.40 |
| BTC Hourly | lstm | LSTM | 1015 | 428 | 587 | 42.17% | 35.42% | 39.38% | 7.83 pp | -159 | 52 | -3.06 |
| BTC Hourly | xgb | XGBoost | 1015 | 415 | 600 | 40.89% | 34.58% | 37.71% | 9.11 pp | -185 | 52 | -3.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 838 | 399 | 439 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| BTC Daily | nn | NN | 838 | 390 | 448 | 46.54% | 45.00% | 45.21% | 3.46 pp | -58 | 48 | -1.21 |
| BTC Daily | transformer | Transformer | 838 | 386 | 452 | 46.06% | 37.92% | 44.17% | 3.94 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 838 | 353 | 485 | 42.12% | 35.83% | 39.79% | 7.88 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 838 | 348 | 490 | 41.53% | 37.08% | 40.83% | 8.47 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 848 | 335 | 513 | 39.50% | 37.50% | 36.88% | 10.50 pp | -178 | 48 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 610 | 294 | 316 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 610 | 291 | 319 | 47.70% | 50.42% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 610 | 285 | 325 | 46.72% | 46.25% | 45.62% | 3.28 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 610 | 261 | 349 | 42.79% | 42.92% | 42.92% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 610 | 261 | 349 | 42.79% | 43.33% | 42.08% | 7.21 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 610 | 259 | 351 | 42.46% | 45.83% | 43.33% | 7.54 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 664 | 313 | 351 | 47.14% | 50.00% | 48.12% | 2.86 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 664 | 311 | 353 | 46.84% | 49.17% | 47.08% | 3.16 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 664 | 309 | 355 | 46.54% | 48.33% | 47.92% | 3.46 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 664 | 276 | 388 | 41.57% | 43.75% | 41.67% | 8.43 pp | -112 | 56 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 664 | 272 | 392 | 40.96% | 44.17% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 664 | 272 | 392 | 40.96% | 43.33% | 40.83% | 9.04 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
