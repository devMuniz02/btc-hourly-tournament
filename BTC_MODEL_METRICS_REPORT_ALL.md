# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T21:14:08.300077+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1298 | 1010 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1174 | 809 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 20:00:00+00:00 | 903 | 571 | 331 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 20:00:00+00:00 | 905 | 625 | 278 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 214 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 571 | 278 | 293 | 48.69% | 47.50% | 47.92% | 1.31 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 571 | 271 | 300 | 47.46% | 51.67% | 48.96% | 2.54 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 799 | 385 | 414 | 48.19% | 46.67% | 47.71% | 1.81 pp | -29 | 46 | -0.63 |
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| BTC Market Hours | transformer | Transformer | 571 | 268 | 303 | 46.94% | 47.08% | 46.67% | 3.06 pp | -35 | 53 | -0.66 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | nn | NN | 625 | 292 | 333 | 46.72% | 47.92% | 48.12% | 3.28 pp | -41 | 53 | -0.77 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 625 | 291 | 334 | 46.56% | 48.75% | 47.29% | 3.44 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 625 | 291 | 334 | 46.56% | 48.33% | 47.50% | 3.44 pp | -43 | 53 | -0.81 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 976 | 464 | 512 | 47.54% | 50.00% | 46.88% | 2.46 pp | -48 | 51 | -0.94 |
| BTC Daily | nn | NN | 799 | 372 | 427 | 46.56% | 45.42% | 45.21% | 3.44 pp | -55 | 46 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| BTC Daily | transformer | Transformer | 799 | 370 | 429 | 46.31% | 39.17% | 46.04% | 3.69 pp | -59 | 46 | -1.28 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| BTC Hourly | transformer | Transformer | 976 | 453 | 523 | 46.41% | 43.75% | 43.54% | 3.59 pp | -70 | 51 | -1.37 |
| BTC Market Hours | lstm | LSTM | 571 | 248 | 323 | 43.43% | 42.92% | 43.75% | 6.57 pp | -75 | 53 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 571 | 245 | 326 | 42.91% | 44.58% | 43.54% | 7.09 pp | -81 | 53 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 571 | 245 | 326 | 42.91% | 45.83% | 43.33% | 7.09 pp | -81 | 53 | -1.53 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 625 | 260 | 365 | 41.60% | 42.92% | 40.62% | 8.40 pp | -105 | 53 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 625 | 258 | 367 | 41.28% | 43.75% | 41.04% | 8.72 pp | -109 | 53 | -2.06 |
| BTC Hourly | rf | RandomForest | 976 | 432 | 544 | 44.26% | 41.67% | 42.92% | 5.74 pp | -112 | 51 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 625 | 254 | 371 | 40.64% | 40.83% | 40.00% | 9.36 pp | -117 | 53 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| BTC Hourly | nn | NN | 976 | 431 | 545 | 44.16% | 41.25% | 42.50% | 5.84 pp | -114 | 51 | -2.24 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 799 | 336 | 463 | 42.05% | 34.17% | 40.00% | 7.95 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| BTC Daily | rf | RandomForest | 799 | 334 | 465 | 41.80% | 37.50% | 41.46% | 8.20 pp | -131 | 46 | -2.85 |
| BTC Hourly | lstm | LSTM | 976 | 415 | 561 | 42.52% | 37.50% | 40.83% | 7.48 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 976 | 403 | 573 | 41.29% | 35.42% | 39.17% | 8.71 pp | -170 | 51 | -3.33 |
| BTC Daily | xgb | XGBoost | 809 | 315 | 494 | 38.94% | 35.00% | 35.62% | 11.06 pp | -179 | 46 | -3.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 976 | 464 | 512 | 47.54% | 50.00% | 46.88% | 2.46 pp | -48 | 51 | -0.94 |
| BTC Hourly | transformer | Transformer | 976 | 453 | 523 | 46.41% | 43.75% | 43.54% | 3.59 pp | -70 | 51 | -1.37 |
| BTC Hourly | rf | RandomForest | 976 | 432 | 544 | 44.26% | 41.67% | 42.92% | 5.74 pp | -112 | 51 | -2.20 |
| BTC Hourly | nn | NN | 976 | 431 | 545 | 44.16% | 41.25% | 42.50% | 5.84 pp | -114 | 51 | -2.24 |
| BTC Hourly | lstm | LSTM | 976 | 415 | 561 | 42.52% | 37.50% | 40.83% | 7.48 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 976 | 403 | 573 | 41.29% | 35.42% | 39.17% | 8.71 pp | -170 | 51 | -3.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 799 | 385 | 414 | 48.19% | 46.67% | 47.71% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 799 | 372 | 427 | 46.56% | 45.42% | 45.21% | 3.44 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 799 | 370 | 429 | 46.31% | 39.17% | 46.04% | 3.69 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 799 | 336 | 463 | 42.05% | 34.17% | 40.00% | 7.95 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 799 | 334 | 465 | 41.80% | 37.50% | 41.46% | 8.20 pp | -131 | 46 | -2.85 |
| BTC Daily | xgb | XGBoost | 809 | 315 | 494 | 38.94% | 35.00% | 35.62% | 11.06 pp | -179 | 46 | -3.89 |

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
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
