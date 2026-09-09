# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T00:04:22.452301+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1300 | 1012 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1176 | 811 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 23:00:00+00:00 | 908 | 573 | 334 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 23:00:00+00:00 | 910 | 627 | 281 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 573 | 279 | 294 | 48.69% | 47.92% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 573 | 273 | 300 | 47.64% | 52.08% | 49.17% | 2.36 pp | -27 | 53 | -0.51 |
| BTC Daily | mlp_sklearn | MLPClassifier | 801 | 386 | 415 | 48.19% | 47.08% | 47.50% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Market Hours | transformer | Transformer | 573 | 269 | 304 | 46.95% | 46.67% | 46.67% | 3.05 pp | -35 | 53 | -0.66 |
| BTC Market Hours Daily | nn | NN | 627 | 293 | 334 | 46.73% | 47.92% | 48.12% | 3.27 pp | -41 | 53 | -0.77 |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 627 | 292 | 335 | 46.57% | 48.75% | 47.08% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 627 | 292 | 335 | 46.57% | 48.75% | 47.29% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 978 | 466 | 512 | 47.65% | 50.00% | 46.88% | 2.35 pp | -46 | 51 | -0.90 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 801 | 373 | 428 | 46.57% | 45.42% | 45.00% | 3.43 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 801 | 371 | 430 | 46.32% | 39.58% | 45.83% | 3.68 pp | -59 | 46 | -1.28 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| BTC Hourly | transformer | Transformer | 978 | 453 | 525 | 46.32% | 43.75% | 43.33% | 3.68 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 573 | 249 | 324 | 43.46% | 43.33% | 43.75% | 6.54 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 573 | 247 | 326 | 43.11% | 45.42% | 43.75% | 6.89 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 573 | 247 | 326 | 43.11% | 46.67% | 43.54% | 6.89 pp | -79 | 53 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 627 | 261 | 366 | 41.63% | 42.92% | 40.62% | 8.37 pp | -105 | 53 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 627 | 259 | 368 | 41.31% | 44.17% | 40.83% | 8.69 pp | -109 | 53 | -2.06 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 627 | 255 | 372 | 40.67% | 41.25% | 40.00% | 9.33 pp | -117 | 53 | -2.21 |
| BTC Hourly | rf | RandomForest | 978 | 432 | 546 | 44.17% | 41.67% | 42.71% | 5.83 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 978 | 431 | 547 | 44.07% | 41.25% | 42.29% | 5.93 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 801 | 337 | 464 | 42.07% | 34.58% | 40.00% | 7.93 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 978 | 416 | 562 | 42.54% | 37.92% | 40.83% | 7.46 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 801 | 334 | 467 | 41.70% | 37.08% | 41.04% | 8.30 pp | -133 | 46 | -2.89 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 978 | 404 | 574 | 41.31% | 35.83% | 38.96% | 8.69 pp | -170 | 51 | -3.33 |
| BTC Daily | xgb | XGBoost | 811 | 316 | 495 | 38.96% | 35.00% | 35.62% | 11.04 pp | -179 | 46 | -3.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 978 | 466 | 512 | 47.65% | 50.00% | 46.88% | 2.35 pp | -46 | 51 | -0.90 |
| BTC Hourly | transformer | Transformer | 978 | 453 | 525 | 46.32% | 43.75% | 43.33% | 3.68 pp | -72 | 51 | -1.41 |
| BTC Hourly | rf | RandomForest | 978 | 432 | 546 | 44.17% | 41.67% | 42.71% | 5.83 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 978 | 431 | 547 | 44.07% | 41.25% | 42.29% | 5.93 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 978 | 416 | 562 | 42.54% | 37.92% | 40.83% | 7.46 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 978 | 404 | 574 | 41.31% | 35.83% | 38.96% | 8.69 pp | -170 | 51 | -3.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 801 | 386 | 415 | 48.19% | 47.08% | 47.50% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 801 | 373 | 428 | 46.57% | 45.42% | 45.00% | 3.43 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 801 | 371 | 430 | 46.32% | 39.58% | 45.83% | 3.68 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 801 | 337 | 464 | 42.07% | 34.58% | 40.00% | 7.93 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 801 | 334 | 467 | 41.70% | 37.08% | 41.04% | 8.30 pp | -133 | 46 | -2.89 |
| BTC Daily | xgb | XGBoost | 811 | 316 | 495 | 38.96% | 35.00% | 35.62% | 11.04 pp | -179 | 46 | -3.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 573 | 279 | 294 | 48.69% | 47.92% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 573 | 273 | 300 | 47.64% | 52.08% | 49.17% | 2.36 pp | -27 | 53 | -0.51 |
| BTC Market Hours | transformer | Transformer | 573 | 269 | 304 | 46.95% | 46.67% | 46.67% | 3.05 pp | -35 | 53 | -0.66 |
| BTC Market Hours | lstm | LSTM | 573 | 249 | 324 | 43.46% | 43.33% | 43.75% | 6.54 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 573 | 247 | 326 | 43.11% | 45.42% | 43.75% | 6.89 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 573 | 247 | 326 | 43.11% | 46.67% | 43.54% | 6.89 pp | -79 | 53 | -1.49 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 627 | 293 | 334 | 46.73% | 47.92% | 48.12% | 3.27 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 627 | 292 | 335 | 46.57% | 48.75% | 47.08% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 627 | 292 | 335 | 46.57% | 48.75% | 47.29% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 627 | 261 | 366 | 41.63% | 42.92% | 40.62% | 8.37 pp | -105 | 53 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 627 | 259 | 368 | 41.31% | 44.17% | 40.83% | 8.69 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 627 | 255 | 372 | 40.67% | 41.25% | 40.00% | 9.33 pp | -117 | 53 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
