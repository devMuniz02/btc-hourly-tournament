# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T01:09:26.715702+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1301 | 1013 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1176 | 811 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 909 | 573 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 911 | 627 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 573 | 279 | 294 | 48.69% | 47.92% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| BTC Market Hours | nn | NN | 573 | 273 | 300 | 47.64% | 52.08% | 49.17% | 2.36 pp | -27 | 53 | -0.51 |
| BTC Market Hours | transformer | Transformer | 573 | 269 | 304 | 46.95% | 46.67% | 46.67% | 3.05 pp | -35 | 53 | -0.66 |
| BTC Daily | mlp_sklearn | MLPClassifier | 801 | 385 | 416 | 48.06% | 46.67% | 47.29% | 1.94 pp | -31 | 46 | -0.67 |
| BTC Market Hours Daily | nn | NN | 627 | 293 | 334 | 46.73% | 47.92% | 48.12% | 3.27 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 627 | 292 | 335 | 46.57% | 48.75% | 47.08% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 627 | 292 | 335 | 46.57% | 48.75% | 47.29% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 801 | 372 | 429 | 46.44% | 45.00% | 44.79% | 3.56 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 801 | 370 | 431 | 46.19% | 39.58% | 45.62% | 3.81 pp | -61 | 46 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 573 | 249 | 324 | 43.46% | 43.33% | 43.75% | 6.54 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 573 | 247 | 326 | 43.11% | 45.42% | 43.75% | 6.89 pp | -79 | 53 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 573 | 247 | 326 | 43.11% | 46.67% | 43.54% | 6.89 pp | -79 | 53 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 627 | 261 | 366 | 41.63% | 42.92% | 40.62% | 8.37 pp | -105 | 53 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 627 | 259 | 368 | 41.31% | 44.17% | 40.83% | 8.69 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 627 | 255 | 372 | 40.67% | 41.25% | 40.00% | 9.33 pp | -117 | 53 | -2.21 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 801 | 337 | 464 | 42.07% | 34.58% | 40.00% | 7.93 pp | -127 | 46 | -2.76 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 801 | 333 | 468 | 41.57% | 37.08% | 40.83% | 8.43 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 811 | 315 | 496 | 38.84% | 35.00% | 35.42% | 11.16 pp | -181 | 46 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 801 | 385 | 416 | 48.06% | 46.67% | 47.29% | 1.94 pp | -31 | 46 | -0.67 |
| BTC Daily | nn | NN | 801 | 372 | 429 | 46.44% | 45.00% | 44.79% | 3.56 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 801 | 370 | 431 | 46.19% | 39.58% | 45.62% | 3.81 pp | -61 | 46 | -1.33 |
| BTC Daily | lstm | LSTM | 801 | 337 | 464 | 42.07% | 34.58% | 40.00% | 7.93 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 801 | 333 | 468 | 41.57% | 37.08% | 40.83% | 8.43 pp | -135 | 46 | -2.93 |
| BTC Daily | xgb | XGBoost | 811 | 315 | 496 | 38.84% | 35.00% | 35.42% | 11.16 pp | -181 | 46 | -3.93 |

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
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
