# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T04:13:32.650059+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1334 | 1046 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1210 | 845 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 969 | 607 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 971 | 661 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T21:00:00+00:00 | 247 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T21:00:00+00:00 | 247 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T21:00:00+00:00 | 247 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T21:00:00+00:00 | 248 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 607 | 292 | 315 | 48.11% | 46.25% | 47.71% | 1.89 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 607 | 289 | 318 | 47.61% | 50.00% | 49.17% | 2.39 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 607 | 284 | 323 | 46.79% | 45.83% | 46.04% | 3.21 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 661 | 310 | 351 | 46.90% | 49.17% | 47.08% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | nn | NN | 661 | 310 | 351 | 46.90% | 48.75% | 47.71% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 835 | 398 | 437 | 47.66% | 45.00% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 661 | 307 | 354 | 46.44% | 48.33% | 47.50% | 3.56 pp | -47 | 56 | -0.84 |
| Consolidated Hourly | rf | RandomForest | 247 | 116 | 131 | 46.96% | 47.08% | 46.96% | 3.04 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 116 | 131 | 46.96% | 47.08% | 46.96% | 3.04 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1012 | 479 | 533 | 47.33% | 48.33% | 45.62% | 2.67 pp | -54 | 52 | -1.04 |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 835 | 388 | 447 | 46.47% | 45.00% | 45.00% | 3.53 pp | -59 | 48 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1012 | 472 | 540 | 46.64% | 45.83% | 44.38% | 3.36 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 835 | 385 | 450 | 46.11% | 37.92% | 44.38% | 3.89 pp | -65 | 48 | -1.35 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 40 | 50 | 44.44% | 44.44% | 44.44% | 5.56 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 247 | 112 | 135 | 45.34% | 45.00% | 45.34% | 4.66 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 112 | 135 | 45.34% | 45.00% | 45.34% | 4.66 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 607 | 260 | 347 | 42.83% | 42.92% | 43.12% | 7.17 pp | -87 | 56 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 607 | 259 | 348 | 42.67% | 42.92% | 42.29% | 7.33 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 607 | 257 | 350 | 42.34% | 45.00% | 43.33% | 7.66 pp | -93 | 56 | -1.66 |
| Consolidated Hourly | transformer | Transformer | 247 | 109 | 138 | 44.13% | 43.75% | 44.13% | 5.87 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 109 | 138 | 44.13% | 43.75% | 44.13% | 5.87 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 661 | 274 | 387 | 41.45% | 42.92% | 41.46% | 8.55 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 661 | 270 | 391 | 40.85% | 43.33% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 661 | 270 | 391 | 40.85% | 42.50% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| Consolidated Hourly | xgb | XGBoost | 247 | 107 | 140 | 43.32% | 43.33% | 43.32% | 6.68 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 107 | 140 | 43.32% | 43.33% | 43.32% | 6.68 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 1012 | 447 | 565 | 44.17% | 42.92% | 41.25% | 5.83 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 1012 | 445 | 567 | 43.97% | 40.83% | 42.92% | 6.03 pp | -122 | 52 | -2.35 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 247 | 104 | 143 | 42.11% | 42.92% | 42.11% | 7.89 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 104 | 143 | 42.11% | 42.92% | 42.11% | 7.89 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 835 | 353 | 482 | 42.28% | 36.25% | 40.21% | 7.72 pp | -129 | 48 | -2.69 |
| BTC Daily | rf | RandomForest | 835 | 346 | 489 | 41.44% | 37.08% | 40.62% | 8.56 pp | -143 | 48 | -2.98 |
| BTC Hourly | lstm | LSTM | 1012 | 428 | 584 | 42.29% | 36.67% | 39.58% | 7.71 pp | -156 | 52 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1012 | 415 | 597 | 41.01% | 34.58% | 38.12% | 8.99 pp | -182 | 52 | -3.50 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 845 | 333 | 512 | 39.41% | 37.50% | 36.67% | 10.59 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1012 | 479 | 533 | 47.33% | 48.33% | 45.62% | 2.67 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1012 | 472 | 540 | 46.64% | 45.83% | 44.38% | 3.36 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1012 | 447 | 565 | 44.17% | 42.92% | 41.25% | 5.83 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1012 | 445 | 567 | 43.97% | 40.83% | 42.92% | 6.03 pp | -122 | 52 | -2.35 |
| BTC Hourly | lstm | LSTM | 1012 | 428 | 584 | 42.29% | 36.67% | 39.58% | 7.71 pp | -156 | 52 | -3.00 |
| BTC Hourly | xgb | XGBoost | 1012 | 415 | 597 | 41.01% | 34.58% | 38.12% | 8.99 pp | -182 | 52 | -3.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 835 | 398 | 437 | 47.66% | 45.00% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Daily | nn | NN | 835 | 388 | 447 | 46.47% | 45.00% | 45.00% | 3.53 pp | -59 | 48 | -1.23 |
| BTC Daily | transformer | Transformer | 835 | 385 | 450 | 46.11% | 37.92% | 44.38% | 3.89 pp | -65 | 48 | -1.35 |
| BTC Daily | lstm | LSTM | 835 | 353 | 482 | 42.28% | 36.25% | 40.21% | 7.72 pp | -129 | 48 | -2.69 |
| BTC Daily | rf | RandomForest | 835 | 346 | 489 | 41.44% | 37.08% | 40.62% | 8.56 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 845 | 333 | 512 | 39.41% | 37.50% | 36.67% | 10.59 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 607 | 292 | 315 | 48.11% | 46.25% | 47.71% | 1.89 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 607 | 289 | 318 | 47.61% | 50.00% | 49.17% | 2.39 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 607 | 284 | 323 | 46.79% | 45.83% | 46.04% | 3.21 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 607 | 260 | 347 | 42.83% | 42.92% | 43.12% | 7.17 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 607 | 259 | 348 | 42.67% | 42.92% | 42.29% | 7.33 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 607 | 257 | 350 | 42.34% | 45.00% | 43.33% | 7.66 pp | -93 | 56 | -1.66 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 661 | 310 | 351 | 46.90% | 49.17% | 47.08% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | nn | NN | 661 | 310 | 351 | 46.90% | 48.75% | 47.71% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 661 | 307 | 354 | 46.44% | 48.33% | 47.50% | 3.56 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 661 | 274 | 387 | 41.45% | 42.92% | 41.46% | 8.55 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 661 | 270 | 391 | 40.85% | 43.33% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 661 | 270 | 391 | 40.85% | 42.50% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 247 | 116 | 131 | 46.96% | 47.08% | 46.96% | 3.04 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | lstm | LSTM | 247 | 112 | 135 | 45.34% | 45.00% | 45.34% | 4.66 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 247 | 109 | 138 | 44.13% | 43.75% | 44.13% | 5.87 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 247 | 107 | 140 | 43.32% | 43.33% | 43.32% | 6.68 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 247 | 104 | 143 | 42.11% | 42.92% | 42.11% | 7.89 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 116 | 131 | 46.96% | 47.08% | 46.96% | 3.04 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 112 | 135 | 45.34% | 45.00% | 45.34% | 4.66 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 109 | 138 | 44.13% | 43.75% | 44.13% | 5.87 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 107 | 140 | 43.32% | 43.33% | 43.32% | 6.68 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 104 | 143 | 42.11% | 42.92% | 42.11% | 7.89 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 40 | 50 | 44.44% | 44.44% | 44.44% | 5.56 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
