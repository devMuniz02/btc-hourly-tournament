# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T20:32:12.830728+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1297 | 1009 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1173 | 808 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 19:00:00+00:00 | 901 | 570 | 330 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 19:00:00+00:00 | 903 | 624 | 277 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 570 | 278 | 292 | 48.77% | 47.50% | 48.12% | 1.23 pp | -14 | 53 | -0.26 |
| Consolidated Hourly | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 570 | 271 | 299 | 47.54% | 51.67% | 49.17% | 2.46 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 798 | 385 | 413 | 48.25% | 47.08% | 47.71% | 1.75 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 570 | 267 | 303 | 46.84% | 46.67% | 46.67% | 3.16 pp | -36 | 53 | -0.68 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | nn | NN | 624 | 291 | 333 | 46.63% | 47.92% | 47.92% | 3.37 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 624 | 291 | 333 | 46.63% | 48.75% | 47.50% | 3.37 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 624 | 290 | 334 | 46.47% | 48.75% | 47.08% | 3.53 pp | -44 | 53 | -0.83 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 975 | 464 | 511 | 47.59% | 50.00% | 46.88% | 2.41 pp | -47 | 51 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 798 | 371 | 427 | 46.49% | 45.00% | 45.00% | 3.51 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 798 | 370 | 428 | 46.37% | 39.58% | 46.04% | 3.63 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 975 | 453 | 522 | 46.46% | 44.17% | 43.75% | 3.54 pp | -69 | 51 | -1.35 |
| BTC Market Hours | lstm | LSTM | 570 | 247 | 323 | 43.33% | 42.50% | 43.75% | 6.67 pp | -76 | 53 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 570 | 245 | 325 | 42.98% | 45.00% | 43.54% | 7.02 pp | -80 | 53 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 570 | 245 | 325 | 42.98% | 46.25% | 43.54% | 7.02 pp | -80 | 53 | -1.51 |
| Consolidated Hourly | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 624 | 260 | 364 | 41.67% | 43.33% | 40.83% | 8.33 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 624 | 257 | 367 | 41.19% | 43.75% | 40.83% | 8.81 pp | -110 | 53 | -2.08 |
| BTC Hourly | rf | RandomForest | 975 | 432 | 543 | 44.31% | 42.08% | 42.92% | 5.69 pp | -111 | 51 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 624 | 254 | 370 | 40.71% | 41.25% | 40.00% | 9.29 pp | -116 | 53 | -2.19 |
| BTC Hourly | nn | NN | 975 | 431 | 544 | 44.21% | 41.25% | 42.50% | 5.79 pp | -113 | 51 | -2.22 |
| Consolidated Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 798 | 336 | 462 | 42.11% | 34.17% | 40.00% | 7.89 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 798 | 334 | 464 | 41.85% | 37.92% | 41.46% | 8.15 pp | -130 | 46 | -2.83 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 975 | 415 | 560 | 42.56% | 37.92% | 41.04% | 7.44 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 975 | 403 | 572 | 41.33% | 35.83% | 39.17% | 8.67 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 808 | 315 | 493 | 38.99% | 35.42% | 35.83% | 11.01 pp | -178 | 46 | -3.87 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 975 | 464 | 511 | 47.59% | 50.00% | 46.88% | 2.41 pp | -47 | 51 | -0.92 |
| BTC Hourly | transformer | Transformer | 975 | 453 | 522 | 46.46% | 44.17% | 43.75% | 3.54 pp | -69 | 51 | -1.35 |
| BTC Hourly | rf | RandomForest | 975 | 432 | 543 | 44.31% | 42.08% | 42.92% | 5.69 pp | -111 | 51 | -2.18 |
| BTC Hourly | nn | NN | 975 | 431 | 544 | 44.21% | 41.25% | 42.50% | 5.79 pp | -113 | 51 | -2.22 |
| BTC Hourly | lstm | LSTM | 975 | 415 | 560 | 42.56% | 37.92% | 41.04% | 7.44 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 975 | 403 | 572 | 41.33% | 35.83% | 39.17% | 8.67 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 798 | 385 | 413 | 48.25% | 47.08% | 47.71% | 1.75 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 798 | 371 | 427 | 46.49% | 45.00% | 45.00% | 3.51 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 798 | 370 | 428 | 46.37% | 39.58% | 46.04% | 3.63 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 798 | 336 | 462 | 42.11% | 34.17% | 40.00% | 7.89 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 798 | 334 | 464 | 41.85% | 37.92% | 41.46% | 8.15 pp | -130 | 46 | -2.83 |
| BTC Daily | xgb | XGBoost | 808 | 315 | 493 | 38.99% | 35.42% | 35.83% | 11.01 pp | -178 | 46 | -3.87 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 570 | 278 | 292 | 48.77% | 47.50% | 48.12% | 1.23 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 570 | 271 | 299 | 47.54% | 51.67% | 49.17% | 2.46 pp | -28 | 53 | -0.53 |
| BTC Market Hours | transformer | Transformer | 570 | 267 | 303 | 46.84% | 46.67% | 46.67% | 3.16 pp | -36 | 53 | -0.68 |
| BTC Market Hours | lstm | LSTM | 570 | 247 | 323 | 43.33% | 42.50% | 43.75% | 6.67 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 570 | 245 | 325 | 42.98% | 45.00% | 43.54% | 7.02 pp | -80 | 53 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 570 | 245 | 325 | 42.98% | 46.25% | 43.54% | 7.02 pp | -80 | 53 | -1.51 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 624 | 291 | 333 | 46.63% | 47.92% | 47.92% | 3.37 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 624 | 291 | 333 | 46.63% | 48.75% | 47.50% | 3.37 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 624 | 290 | 334 | 46.47% | 48.75% | 47.08% | 3.53 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 624 | 260 | 364 | 41.67% | 43.33% | 40.83% | 8.33 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 624 | 257 | 367 | 41.19% | 43.75% | 40.83% | 8.81 pp | -110 | 53 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 624 | 254 | 370 | 40.71% | 41.25% | 40.00% | 9.29 pp | -116 | 53 | -2.19 |

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
