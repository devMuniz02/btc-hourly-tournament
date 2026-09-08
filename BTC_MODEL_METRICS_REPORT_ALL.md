# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T15:56:34.403131+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1170 | 805 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 14:00:00+00:00 | 893 | 567 | 325 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 14:00:00+00:00 | 895 | 621 | 272 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 567 | 276 | 291 | 48.68% | 47.08% | 47.92% | 1.32 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 567 | 269 | 298 | 47.44% | 51.25% | 48.96% | 2.56 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 795 | 384 | 411 | 48.30% | 47.08% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 567 | 267 | 300 | 47.09% | 47.08% | 47.08% | 2.91 pp | -33 | 53 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 621 | 290 | 331 | 46.70% | 48.75% | 47.71% | 3.30 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | nn | NN | 621 | 289 | 332 | 46.54% | 47.08% | 47.71% | 3.46 pp | -43 | 53 | -0.81 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 621 | 288 | 333 | 46.38% | 48.33% | 47.08% | 3.62 pp | -45 | 53 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 973 | 462 | 511 | 47.48% | 49.17% | 46.46% | 2.52 pp | -49 | 51 | -0.96 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| BTC Daily | transformer | Transformer | 795 | 369 | 426 | 46.42% | 40.00% | 46.25% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | nn | NN | 795 | 368 | 427 | 46.29% | 44.58% | 44.79% | 3.71 pp | -59 | 46 | -1.28 |
| BTC Hourly | transformer | Transformer | 973 | 453 | 520 | 46.56% | 44.58% | 43.75% | 3.44 pp | -67 | 51 | -1.31 |
| BTC Market Hours | lstm | LSTM | 567 | 246 | 321 | 43.39% | 42.50% | 43.75% | 6.61 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 567 | 245 | 322 | 43.21% | 45.83% | 43.75% | 6.79 pp | -77 | 53 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 567 | 244 | 323 | 43.03% | 46.67% | 43.54% | 6.97 pp | -79 | 53 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 621 | 259 | 362 | 41.71% | 43.75% | 40.62% | 8.29 pp | -103 | 53 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 621 | 256 | 365 | 41.22% | 44.17% | 40.83% | 8.78 pp | -109 | 53 | -2.06 |
| BTC Hourly | rf | RandomForest | 973 | 431 | 542 | 44.30% | 42.08% | 42.71% | 5.70 pp | -111 | 51 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 621 | 252 | 369 | 40.58% | 40.42% | 39.79% | 9.42 pp | -117 | 53 | -2.21 |
| BTC Hourly | nn | NN | 973 | 430 | 543 | 44.19% | 41.25% | 42.50% | 5.81 pp | -113 | 51 | -2.22 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 795 | 334 | 461 | 42.01% | 33.75% | 40.00% | 7.99 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 973 | 414 | 559 | 42.55% | 37.92% | 41.25% | 7.45 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 795 | 331 | 464 | 41.64% | 37.50% | 41.25% | 8.36 pp | -133 | 46 | -2.89 |
| BTC Hourly | xgb | XGBoost | 973 | 401 | 572 | 41.21% | 35.83% | 38.75% | 8.79 pp | -171 | 51 | -3.35 |
| BTC Daily | xgb | XGBoost | 805 | 313 | 492 | 38.88% | 35.42% | 35.62% | 11.12 pp | -179 | 46 | -3.89 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 795 | 384 | 411 | 48.30% | 47.08% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 795 | 369 | 426 | 46.42% | 40.00% | 46.25% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | nn | NN | 795 | 368 | 427 | 46.29% | 44.58% | 44.79% | 3.71 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 795 | 334 | 461 | 42.01% | 33.75% | 40.00% | 7.99 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 795 | 331 | 464 | 41.64% | 37.50% | 41.25% | 8.36 pp | -133 | 46 | -2.89 |
| BTC Daily | xgb | XGBoost | 805 | 313 | 492 | 38.88% | 35.42% | 35.62% | 11.12 pp | -179 | 46 | -3.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 567 | 276 | 291 | 48.68% | 47.08% | 47.92% | 1.32 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 567 | 269 | 298 | 47.44% | 51.25% | 48.96% | 2.56 pp | -29 | 53 | -0.55 |
| BTC Market Hours | transformer | Transformer | 567 | 267 | 300 | 47.09% | 47.08% | 47.08% | 2.91 pp | -33 | 53 | -0.62 |
| BTC Market Hours | lstm | LSTM | 567 | 246 | 321 | 43.39% | 42.50% | 43.75% | 6.61 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 567 | 245 | 322 | 43.21% | 45.83% | 43.75% | 6.79 pp | -77 | 53 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 567 | 244 | 323 | 43.03% | 46.67% | 43.54% | 6.97 pp | -79 | 53 | -1.49 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 621 | 290 | 331 | 46.70% | 48.75% | 47.71% | 3.30 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | nn | NN | 621 | 289 | 332 | 46.54% | 47.08% | 47.71% | 3.46 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 621 | 288 | 333 | 46.38% | 48.33% | 47.08% | 3.62 pp | -45 | 53 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 621 | 259 | 362 | 41.71% | 43.75% | 40.62% | 8.29 pp | -103 | 53 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 621 | 256 | 365 | 41.22% | 44.17% | 40.83% | 8.78 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 621 | 252 | 369 | 40.58% | 40.42% | 39.79% | 9.42 pp | -117 | 53 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
