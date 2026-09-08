# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T22:10:14.180125+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1299 | 1011 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1174 | 809 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 21:00:00+00:00 | 904 | 571 | 332 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 21:00:00+00:00 | 906 | 625 | 279 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 571 | 278 | 293 | 48.69% | 47.50% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 571 | 271 | 300 | 47.46% | 51.67% | 48.96% | 2.54 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 799 | 385 | 414 | 48.19% | 47.08% | 47.71% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Market Hours | transformer | Transformer | 571 | 268 | 303 | 46.94% | 47.08% | 46.67% | 3.06 pp | -35 | 53 | -0.66 |
| BTC Market Hours Daily | nn | NN | 625 | 292 | 333 | 46.72% | 47.92% | 48.12% | 3.28 pp | -41 | 53 | -0.77 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 625 | 291 | 334 | 46.56% | 48.75% | 47.29% | 3.44 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 625 | 291 | 334 | 46.56% | 48.33% | 47.50% | 3.44 pp | -43 | 53 | -0.81 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 977 | 465 | 512 | 47.59% | 50.00% | 46.88% | 2.41 pp | -47 | 51 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 799 | 371 | 428 | 46.43% | 45.42% | 45.00% | 3.57 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 799 | 369 | 430 | 46.18% | 39.17% | 45.83% | 3.82 pp | -61 | 46 | -1.33 |
| BTC Hourly | transformer | Transformer | 977 | 453 | 524 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 571 | 248 | 323 | 43.43% | 42.92% | 43.75% | 6.57 pp | -75 | 53 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 571 | 245 | 326 | 42.91% | 44.58% | 43.54% | 7.09 pp | -81 | 53 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 571 | 245 | 326 | 42.91% | 45.83% | 43.33% | 7.09 pp | -81 | 53 | -1.53 |
| Consolidated Hourly | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 625 | 260 | 365 | 41.60% | 42.92% | 40.62% | 8.40 pp | -105 | 53 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 625 | 258 | 367 | 41.28% | 43.75% | 41.04% | 8.72 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 625 | 254 | 371 | 40.64% | 40.83% | 40.00% | 9.36 pp | -117 | 53 | -2.21 |
| BTC Hourly | rf | RandomForest | 977 | 432 | 545 | 44.22% | 41.67% | 42.71% | 5.78 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 977 | 431 | 546 | 44.11% | 41.25% | 42.50% | 5.89 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 799 | 337 | 462 | 42.18% | 34.58% | 40.21% | 7.82 pp | -125 | 46 | -2.72 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 977 | 415 | 562 | 42.48% | 37.50% | 40.83% | 7.52 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 799 | 333 | 466 | 41.68% | 37.08% | 41.25% | 8.32 pp | -133 | 46 | -2.89 |
| BTC Hourly | xgb | XGBoost | 977 | 403 | 574 | 41.25% | 35.42% | 38.96% | 8.75 pp | -171 | 51 | -3.35 |
| BTC Daily | xgb | XGBoost | 809 | 314 | 495 | 38.81% | 34.58% | 35.42% | 11.19 pp | -181 | 46 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 977 | 465 | 512 | 47.59% | 50.00% | 46.88% | 2.41 pp | -47 | 51 | -0.92 |
| BTC Hourly | transformer | Transformer | 977 | 453 | 524 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Hourly | rf | RandomForest | 977 | 432 | 545 | 44.22% | 41.67% | 42.71% | 5.78 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 977 | 431 | 546 | 44.11% | 41.25% | 42.50% | 5.89 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 977 | 415 | 562 | 42.48% | 37.50% | 40.83% | 7.52 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 977 | 403 | 574 | 41.25% | 35.42% | 38.96% | 8.75 pp | -171 | 51 | -3.35 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 799 | 385 | 414 | 48.19% | 47.08% | 47.71% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 799 | 371 | 428 | 46.43% | 45.42% | 45.00% | 3.57 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 799 | 369 | 430 | 46.18% | 39.17% | 45.83% | 3.82 pp | -61 | 46 | -1.33 |
| BTC Daily | lstm | LSTM | 799 | 337 | 462 | 42.18% | 34.58% | 40.21% | 7.82 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 799 | 333 | 466 | 41.68% | 37.08% | 41.25% | 8.32 pp | -133 | 46 | -2.89 |
| BTC Daily | xgb | XGBoost | 809 | 314 | 495 | 38.81% | 34.58% | 35.42% | 11.19 pp | -181 | 46 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 571 | 278 | 293 | 48.69% | 47.50% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 571 | 271 | 300 | 47.46% | 51.67% | 48.96% | 2.54 pp | -29 | 53 | -0.55 |
| BTC Market Hours | transformer | Transformer | 571 | 268 | 303 | 46.94% | 47.08% | 46.67% | 3.06 pp | -35 | 53 | -0.66 |
| BTC Market Hours | lstm | LSTM | 571 | 248 | 323 | 43.43% | 42.92% | 43.75% | 6.57 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 571 | 245 | 326 | 42.91% | 44.58% | 43.54% | 7.09 pp | -81 | 53 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 571 | 245 | 326 | 42.91% | 45.83% | 43.33% | 7.09 pp | -81 | 53 | -1.53 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 625 | 292 | 333 | 46.72% | 47.92% | 48.12% | 3.28 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 625 | 291 | 334 | 46.56% | 48.75% | 47.29% | 3.44 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 625 | 291 | 334 | 46.56% | 48.33% | 47.50% | 3.44 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 625 | 260 | 365 | 41.60% | 42.92% | 40.62% | 8.40 pp | -105 | 53 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 625 | 258 | 367 | 41.28% | 43.75% | 41.04% | 8.72 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 625 | 254 | 371 | 40.64% | 40.83% | 40.00% | 9.36 pp | -117 | 53 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
