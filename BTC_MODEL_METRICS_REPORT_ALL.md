# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T20:58:23.849445+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1092 | 727 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 19:00:00+00:00 | 755 | 489 | 265 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 19:00:00+00:00 | 757 | 543 | 212 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 137 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 137 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 30 | 107 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 14:00:00+00:00 | 137 | 30 | 107 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 489 | 235 | 254 | 48.06% | 43.75% | 47.92% | 1.94 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 717 | 346 | 371 | 48.26% | 46.25% | 48.33% | 1.74 pp | -25 | 43 | -0.58 |
| BTC Market Hours | nn | NN | 489 | 229 | 260 | 46.83% | 48.75% | 47.29% | 3.17 pp | -31 | 47 | -0.66 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 489 | 228 | 261 | 46.63% | 42.50% | 47.08% | 3.37 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 717 | 341 | 376 | 47.56% | 45.42% | 49.79% | 2.44 pp | -35 | 43 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 543 | 252 | 291 | 46.41% | 49.58% | 47.50% | 3.59 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 895 | 427 | 468 | 47.71% | 50.42% | 48.33% | 2.29 pp | -41 | 47 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 543 | 249 | 294 | 45.86% | 47.92% | 47.08% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 543 | 249 | 294 | 45.86% | 44.58% | 46.88% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 895 | 424 | 471 | 47.37% | 48.33% | 47.08% | 2.63 pp | -47 | 47 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| BTC Daily | nn | NN | 717 | 331 | 386 | 46.16% | 43.33% | 47.71% | 3.84 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 489 | 211 | 278 | 43.15% | 40.83% | 43.12% | 6.85 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 489 | 210 | 279 | 42.94% | 42.50% | 43.12% | 7.06 pp | -69 | 47 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 489 | 199 | 290 | 40.70% | 40.00% | 40.62% | 9.30 pp | -91 | 47 | -1.94 |
| BTC Market Hours Daily | rf | RandomForest | 543 | 224 | 319 | 41.25% | 41.67% | 41.25% | 8.75 pp | -95 | 47 | -2.02 |
| BTC Hourly | nn | NN | 895 | 399 | 496 | 44.58% | 45.00% | 42.50% | 5.42 pp | -97 | 47 | -2.06 |
| BTC Hourly | rf | RandomForest | 895 | 399 | 496 | 44.58% | 45.42% | 44.17% | 5.42 pp | -97 | 47 | -2.06 |
| Consolidated Hourly | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |
| BTC Daily | lstm | LSTM | 717 | 310 | 407 | 43.24% | 37.92% | 42.29% | 6.76 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 543 | 218 | 325 | 40.15% | 37.92% | 40.62% | 9.85 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 543 | 216 | 327 | 39.78% | 39.58% | 39.38% | 10.22 pp | -111 | 47 | -2.36 |
| BTC Daily | rf | RandomForest | 717 | 305 | 412 | 42.54% | 40.42% | 43.33% | 7.46 pp | -107 | 43 | -2.49 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 895 | 383 | 512 | 42.79% | 39.58% | 42.29% | 7.21 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 895 | 377 | 518 | 42.12% | 42.92% | 42.08% | 7.88 pp | -141 | 47 | -3.00 |
| BTC Daily | xgb | XGBoost | 727 | 287 | 440 | 39.48% | 35.00% | 38.54% | 10.52 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 717 | 346 | 371 | 48.26% | 46.25% | 48.33% | 1.74 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 717 | 341 | 376 | 47.56% | 45.42% | 49.79% | 2.44 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 717 | 331 | 386 | 46.16% | 43.33% | 47.71% | 3.84 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 717 | 310 | 407 | 43.24% | 37.92% | 42.29% | 6.76 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 717 | 305 | 412 | 42.54% | 40.42% | 43.33% | 7.46 pp | -107 | 43 | -2.49 |
| BTC Daily | xgb | XGBoost | 727 | 287 | 440 | 39.48% | 35.00% | 38.54% | 10.52 pp | -153 | 43 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 489 | 235 | 254 | 48.06% | 43.75% | 47.92% | 1.94 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 489 | 229 | 260 | 46.83% | 48.75% | 47.29% | 3.17 pp | -31 | 47 | -0.66 |
| BTC Market Hours | transformer | Transformer | 489 | 228 | 261 | 46.63% | 42.50% | 47.08% | 3.37 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 489 | 211 | 278 | 43.15% | 40.83% | 43.12% | 6.85 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 489 | 210 | 279 | 42.94% | 42.50% | 43.12% | 7.06 pp | -69 | 47 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 489 | 199 | 290 | 40.70% | 40.00% | 40.62% | 9.30 pp | -91 | 47 | -1.94 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 543 | 252 | 291 | 46.41% | 49.58% | 47.50% | 3.59 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 543 | 249 | 294 | 45.86% | 47.92% | 47.08% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 543 | 249 | 294 | 45.86% | 44.58% | 46.88% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 543 | 224 | 319 | 41.25% | 41.67% | 41.25% | 8.75 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 543 | 218 | 325 | 40.15% | 37.92% | 40.62% | 9.85 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 543 | 216 | 327 | 39.78% | 39.58% | 39.38% | 10.22 pp | -111 | 47 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 71 | 66 | 51.82% | 51.82% | 51.82% | 1.82 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 64 | 73 | 46.72% | 46.72% | 46.72% | 3.28 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 60 | 77 | 43.80% | 43.80% | 43.80% | 6.20 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
