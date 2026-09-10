# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T17:04:13.906285+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1327 | 1039 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1203 | 838 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 16:00:00+00:00 | 954 | 600 | 353 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 16:00:00+00:00 | 956 | 654 | 300 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 241 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 600 | 289 | 311 | 48.17% | 46.25% | 47.08% | 1.83 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 600 | 287 | 313 | 47.83% | 51.25% | 49.38% | 2.17 pp | -26 | 56 | -0.46 |
| BTC Market Hours | transformer | Transformer | 600 | 282 | 318 | 47.00% | 46.25% | 46.25% | 3.00 pp | -36 | 56 | -0.64 |
| BTC Market Hours Daily | nn | NN | 654 | 306 | 348 | 46.79% | 47.92% | 47.50% | 3.21 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 654 | 305 | 349 | 46.64% | 47.92% | 47.08% | 3.36 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 654 | 305 | 349 | 46.64% | 48.33% | 48.12% | 3.36 pp | -44 | 55 | -0.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 828 | 395 | 433 | 47.71% | 45.00% | 46.25% | 2.29 pp | -38 | 47 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1005 | 476 | 529 | 47.36% | 48.33% | 45.83% | 2.64 pp | -53 | 52 | -1.02 |
| Consolidated Hourly | rf | RandomForest | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 15 | -1.07 |
| BTC Daily | nn | NN | 828 | 386 | 442 | 46.62% | 45.42% | 45.42% | 3.38 pp | -56 | 47 | -1.19 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 828 | 383 | 445 | 46.26% | 38.33% | 45.00% | 3.74 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1005 | 468 | 537 | 46.57% | 46.25% | 44.79% | 3.43 pp | -69 | 52 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 15 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 15 | -1.47 |
| BTC Market Hours | lstm | LSTM | 600 | 257 | 343 | 42.83% | 42.08% | 42.71% | 7.17 pp | -86 | 56 | -1.54 |
| BTC Market Hours | rf | RandomForest | 600 | 256 | 344 | 42.67% | 42.50% | 42.08% | 7.33 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 600 | 255 | 345 | 42.50% | 45.00% | 43.33% | 7.50 pp | -90 | 56 | -1.61 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 654 | 270 | 384 | 41.28% | 41.67% | 41.04% | 8.72 pp | -114 | 55 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 240 | 104 | 136 | 43.33% | 43.33% | 43.33% | 6.67 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 240 | 104 | 136 | 43.33% | 43.33% | 43.33% | 6.67 pp | -32 | 15 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 654 | 268 | 386 | 40.98% | 42.92% | 40.62% | 9.02 pp | -118 | 55 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 654 | 268 | 386 | 40.98% | 42.50% | 40.62% | 9.02 pp | -118 | 55 | -2.15 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| BTC Hourly | nn | NN | 1005 | 443 | 562 | 44.08% | 42.50% | 41.46% | 5.92 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1005 | 443 | 562 | 44.08% | 42.08% | 43.33% | 5.92 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 240 | 100 | 140 | 41.67% | 41.67% | 41.67% | 8.33 pp | -40 | 15 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 240 | 100 | 140 | 41.67% | 41.67% | 41.67% | 8.33 pp | -40 | 15 | -2.67 |
| BTC Daily | lstm | LSTM | 828 | 348 | 480 | 42.03% | 35.00% | 40.00% | 7.97 pp | -132 | 47 | -2.81 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 1005 | 425 | 580 | 42.29% | 36.67% | 40.00% | 7.71 pp | -155 | 52 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| BTC Daily | rf | RandomForest | 828 | 343 | 485 | 41.43% | 37.08% | 40.42% | 8.57 pp | -142 | 47 | -3.02 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1005 | 414 | 591 | 41.19% | 36.25% | 38.75% | 8.81 pp | -177 | 52 | -3.40 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 838 | 330 | 508 | 39.38% | 37.08% | 36.67% | 10.62 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1005 | 476 | 529 | 47.36% | 48.33% | 45.83% | 2.64 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1005 | 468 | 537 | 46.57% | 46.25% | 44.79% | 3.43 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1005 | 443 | 562 | 44.08% | 42.50% | 41.46% | 5.92 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1005 | 443 | 562 | 44.08% | 42.08% | 43.33% | 5.92 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 1005 | 425 | 580 | 42.29% | 36.67% | 40.00% | 7.71 pp | -155 | 52 | -2.98 |
| BTC Hourly | xgb | XGBoost | 1005 | 414 | 591 | 41.19% | 36.25% | 38.75% | 8.81 pp | -177 | 52 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 828 | 395 | 433 | 47.71% | 45.00% | 46.25% | 2.29 pp | -38 | 47 | -0.81 |
| BTC Daily | nn | NN | 828 | 386 | 442 | 46.62% | 45.42% | 45.42% | 3.38 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 828 | 383 | 445 | 46.26% | 38.33% | 45.00% | 3.74 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 828 | 348 | 480 | 42.03% | 35.00% | 40.00% | 7.97 pp | -132 | 47 | -2.81 |
| BTC Daily | rf | RandomForest | 828 | 343 | 485 | 41.43% | 37.08% | 40.42% | 8.57 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 838 | 330 | 508 | 39.38% | 37.08% | 36.67% | 10.62 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 600 | 289 | 311 | 48.17% | 46.25% | 47.08% | 1.83 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 600 | 287 | 313 | 47.83% | 51.25% | 49.38% | 2.17 pp | -26 | 56 | -0.46 |
| BTC Market Hours | transformer | Transformer | 600 | 282 | 318 | 47.00% | 46.25% | 46.25% | 3.00 pp | -36 | 56 | -0.64 |
| BTC Market Hours | lstm | LSTM | 600 | 257 | 343 | 42.83% | 42.08% | 42.71% | 7.17 pp | -86 | 56 | -1.54 |
| BTC Market Hours | rf | RandomForest | 600 | 256 | 344 | 42.67% | 42.50% | 42.08% | 7.33 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 600 | 255 | 345 | 42.50% | 45.00% | 43.33% | 7.50 pp | -90 | 56 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 654 | 306 | 348 | 46.79% | 47.92% | 47.50% | 3.21 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 654 | 305 | 349 | 46.64% | 47.92% | 47.08% | 3.36 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 654 | 305 | 349 | 46.64% | 48.33% | 48.12% | 3.36 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 654 | 270 | 384 | 41.28% | 41.67% | 41.04% | 8.72 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 654 | 268 | 386 | 40.98% | 42.92% | 40.62% | 9.02 pp | -118 | 55 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 654 | 268 | 386 | 40.98% | 42.50% | 40.62% | 9.02 pp | -118 | 55 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 15 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 240 | 104 | 136 | 43.33% | 43.33% | 43.33% | 6.67 pp | -32 | 15 | -2.13 |
| Consolidated Hourly | nn | NN | 240 | 100 | 140 | 41.67% | 41.67% | 41.67% | 8.33 pp | -40 | 15 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 240 | 109 | 131 | 45.42% | 45.42% | 45.42% | 4.58 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 240 | 104 | 136 | 43.33% | 43.33% | 43.33% | 6.67 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 240 | 100 | 140 | 41.67% | 41.67% | 41.67% | 8.33 pp | -40 | 15 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
