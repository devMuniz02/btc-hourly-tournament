# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T21:52:39.067898+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1217 | 929 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1093 | 728 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 20:00:00+00:00 | 757 | 490 | 266 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 20:00:00+00:00 | 759 | 544 | 213 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 139 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 140 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 490 | 236 | 254 | 48.16% | 44.17% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 718 | 347 | 371 | 48.33% | 46.67% | 48.33% | 1.67 pp | -24 | 43 | -0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| BTC Market Hours | nn | NN | 490 | 230 | 260 | 46.94% | 49.17% | 47.50% | 3.06 pp | -30 | 47 | -0.64 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 490 | 229 | 261 | 46.73% | 42.92% | 47.29% | 3.27 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 718 | 342 | 376 | 47.63% | 45.42% | 50.00% | 2.37 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 544 | 252 | 292 | 46.32% | 49.17% | 47.50% | 3.68 pp | -40 | 47 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 895 | 427 | 468 | 47.71% | 50.42% | 48.33% | 2.29 pp | -41 | 47 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 544 | 249 | 295 | 45.77% | 47.50% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 544 | 249 | 295 | 45.77% | 44.17% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 895 | 424 | 471 | 47.37% | 48.33% | 47.08% | 2.63 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 718 | 332 | 386 | 46.24% | 43.33% | 47.71% | 3.76 pp | -54 | 43 | -1.26 |
| Consolidated Hourly | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 490 | 211 | 279 | 43.06% | 40.83% | 43.12% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 490 | 211 | 279 | 43.06% | 42.92% | 43.33% | 6.94 pp | -68 | 47 | -1.45 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 490 | 200 | 290 | 40.82% | 40.00% | 40.83% | 9.18 pp | -90 | 47 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 544 | 225 | 319 | 41.36% | 41.67% | 41.25% | 8.64 pp | -94 | 47 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 895 | 399 | 496 | 44.58% | 45.00% | 42.50% | 5.42 pp | -97 | 47 | -2.06 |
| BTC Hourly | rf | RandomForest | 895 | 399 | 496 | 44.58% | 45.42% | 44.17% | 5.42 pp | -97 | 47 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 544 | 219 | 325 | 40.26% | 37.92% | 40.83% | 9.74 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 718 | 310 | 408 | 43.18% | 37.50% | 42.08% | 6.82 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 544 | 217 | 327 | 39.89% | 40.00% | 39.38% | 10.11 pp | -110 | 47 | -2.34 |
| BTC Daily | rf | RandomForest | 718 | 306 | 412 | 42.62% | 40.42% | 43.54% | 7.38 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 895 | 383 | 512 | 42.79% | 39.58% | 42.29% | 7.21 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 895 | 377 | 518 | 42.12% | 42.92% | 42.08% | 7.88 pp | -141 | 47 | -3.00 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 728 | 288 | 440 | 39.56% | 35.42% | 38.54% | 10.44 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 895 | 427 | 468 | 47.71% | 50.42% | 48.33% | 2.29 pp | -41 | 47 | -0.87 |
| BTC Hourly | transformer | Transformer | 895 | 424 | 471 | 47.37% | 48.33% | 47.08% | 2.63 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 895 | 399 | 496 | 44.58% | 45.00% | 42.50% | 5.42 pp | -97 | 47 | -2.06 |
| BTC Hourly | rf | RandomForest | 895 | 399 | 496 | 44.58% | 45.42% | 44.17% | 5.42 pp | -97 | 47 | -2.06 |
| BTC Hourly | lstm | LSTM | 895 | 383 | 512 | 42.79% | 39.58% | 42.29% | 7.21 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 895 | 377 | 518 | 42.12% | 42.92% | 42.08% | 7.88 pp | -141 | 47 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 718 | 347 | 371 | 48.33% | 46.67% | 48.33% | 1.67 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 718 | 342 | 376 | 47.63% | 45.42% | 50.00% | 2.37 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 718 | 332 | 386 | 46.24% | 43.33% | 47.71% | 3.76 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 718 | 310 | 408 | 43.18% | 37.50% | 42.08% | 6.82 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 718 | 306 | 412 | 42.62% | 40.42% | 43.54% | 7.38 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 728 | 288 | 440 | 39.56% | 35.42% | 38.54% | 10.44 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 490 | 236 | 254 | 48.16% | 44.17% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 490 | 230 | 260 | 46.94% | 49.17% | 47.50% | 3.06 pp | -30 | 47 | -0.64 |
| BTC Market Hours | transformer | Transformer | 490 | 229 | 261 | 46.73% | 42.92% | 47.29% | 3.27 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 490 | 211 | 279 | 43.06% | 40.83% | 43.12% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 490 | 211 | 279 | 43.06% | 42.92% | 43.33% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 490 | 200 | 290 | 40.82% | 40.00% | 40.83% | 9.18 pp | -90 | 47 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 544 | 252 | 292 | 46.32% | 49.17% | 47.50% | 3.68 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 544 | 249 | 295 | 45.77% | 47.50% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 544 | 249 | 295 | 45.77% | 44.17% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 544 | 225 | 319 | 41.36% | 41.67% | 41.25% | 8.64 pp | -94 | 47 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 544 | 219 | 325 | 40.26% | 37.92% | 40.83% | 9.74 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 544 | 217 | 327 | 39.89% | 40.00% | 39.38% | 10.11 pp | -110 | 47 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 63 | 76 | 45.32% | 45.32% | 45.32% | 4.68 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 62 | 77 | 44.60% | 44.60% | 44.60% | 5.40 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
