# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T14:59:23.193652+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1294 | 1006 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1170 | 805 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 892 | 567 | 324 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 894 | 621 | 271 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 209 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T14:00:00+00:00 | 210 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 567 | 276 | 291 | 48.68% | 47.08% | 47.92% | 1.32 pp | -15 | 53 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| BTC Market Hours | nn | NN | 567 | 269 | 298 | 47.44% | 51.25% | 48.96% | 2.56 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 795 | 384 | 411 | 48.30% | 46.67% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 567 | 267 | 300 | 47.09% | 47.08% | 47.08% | 2.91 pp | -33 | 53 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 621 | 290 | 331 | 46.70% | 48.75% | 47.71% | 3.30 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | nn | NN | 621 | 289 | 332 | 46.54% | 47.08% | 47.71% | 3.46 pp | -43 | 53 | -0.81 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 621 | 288 | 333 | 46.38% | 48.33% | 47.08% | 3.62 pp | -45 | 53 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 972 | 461 | 511 | 47.43% | 48.75% | 46.25% | 2.57 pp | -50 | 50 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| BTC Daily | transformer | Transformer | 795 | 370 | 425 | 46.54% | 40.42% | 46.46% | 3.46 pp | -55 | 46 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 795 | 369 | 426 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Hourly | transformer | Transformer | 972 | 453 | 519 | 46.60% | 44.58% | 43.96% | 3.40 pp | -66 | 50 | -1.32 |
| Consolidated Hourly | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| BTC Market Hours | lstm | LSTM | 567 | 246 | 321 | 43.39% | 42.50% | 43.75% | 6.61 pp | -75 | 53 | -1.42 |
| BTC Market Hours | rf | RandomForest | 567 | 245 | 322 | 43.21% | 45.83% | 43.75% | 6.79 pp | -77 | 53 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 567 | 244 | 323 | 43.03% | 46.67% | 43.54% | 6.97 pp | -79 | 53 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 621 | 259 | 362 | 41.71% | 43.75% | 40.62% | 8.29 pp | -103 | 53 | -1.94 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 621 | 256 | 365 | 41.22% | 44.17% | 40.83% | 8.78 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 621 | 252 | 369 | 40.58% | 40.42% | 39.79% | 9.42 pp | -117 | 53 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 972 | 430 | 542 | 44.24% | 41.67% | 42.71% | 5.76 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 972 | 429 | 543 | 44.14% | 40.83% | 42.50% | 5.86 pp | -114 | 50 | -2.28 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 795 | 334 | 461 | 42.01% | 34.17% | 40.00% | 7.99 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Daily | rf | RandomForest | 795 | 332 | 463 | 41.76% | 37.92% | 41.46% | 8.24 pp | -131 | 46 | -2.85 |
| BTC Hourly | lstm | LSTM | 972 | 413 | 559 | 42.49% | 37.50% | 41.04% | 7.51 pp | -146 | 50 | -2.92 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 972 | 401 | 571 | 41.26% | 35.83% | 38.75% | 8.74 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 805 | 314 | 491 | 39.01% | 35.42% | 35.83% | 10.99 pp | -177 | 46 | -3.85 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 972 | 461 | 511 | 47.43% | 48.75% | 46.25% | 2.57 pp | -50 | 50 | -1.00 |
| BTC Hourly | transformer | Transformer | 972 | 453 | 519 | 46.60% | 44.58% | 43.96% | 3.40 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 972 | 430 | 542 | 44.24% | 41.67% | 42.71% | 5.76 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 972 | 429 | 543 | 44.14% | 40.83% | 42.50% | 5.86 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 972 | 413 | 559 | 42.49% | 37.50% | 41.04% | 7.51 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 972 | 401 | 571 | 41.26% | 35.83% | 38.75% | 8.74 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 795 | 384 | 411 | 48.30% | 46.67% | 47.92% | 1.70 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 795 | 370 | 425 | 46.54% | 40.42% | 46.46% | 3.46 pp | -55 | 46 | -1.20 |
| BTC Daily | nn | NN | 795 | 369 | 426 | 46.42% | 45.00% | 45.00% | 3.58 pp | -57 | 46 | -1.24 |
| BTC Daily | lstm | LSTM | 795 | 334 | 461 | 42.01% | 34.17% | 40.00% | 7.99 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 795 | 332 | 463 | 41.76% | 37.92% | 41.46% | 8.24 pp | -131 | 46 | -2.85 |
| BTC Daily | xgb | XGBoost | 805 | 314 | 491 | 39.01% | 35.42% | 35.83% | 10.99 pp | -177 | 46 | -3.85 |

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
| Consolidated Hourly | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 101 | 108 | 48.33% | 48.33% | 48.33% | 1.67 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 95 | 114 | 45.45% | 45.45% | 45.45% | 4.55 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 89 | 120 | 42.58% | 42.58% | 42.58% | 7.42 pp | -31 | 14 | -2.21 |

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
