# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T15:04:26.034612+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1326 | 1038 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1201 | 836 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 950 | 598 | 351 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 952 | 652 | 298 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 598 | 288 | 310 | 48.16% | 46.25% | 47.29% | 1.84 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 598 | 287 | 311 | 47.99% | 51.67% | 49.79% | 2.01 pp | -24 | 55 | -0.44 |
| BTC Market Hours | transformer | Transformer | 598 | 280 | 318 | 46.82% | 45.83% | 46.25% | 3.18 pp | -38 | 55 | -0.69 |
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | nn | NN | 652 | 305 | 347 | 46.78% | 48.33% | 47.50% | 3.22 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 652 | 304 | 348 | 46.63% | 47.92% | 47.08% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 652 | 304 | 348 | 46.63% | 47.92% | 47.92% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 826 | 394 | 432 | 47.70% | 45.00% | 46.25% | 2.30 pp | -38 | 47 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1004 | 475 | 529 | 47.31% | 47.92% | 45.83% | 2.69 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 826 | 385 | 441 | 46.61% | 45.42% | 45.42% | 3.39 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1004 | 468 | 536 | 46.61% | 46.25% | 45.00% | 3.39 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 826 | 381 | 445 | 46.13% | 37.92% | 44.79% | 3.87 pp | -64 | 47 | -1.36 |
| BTC Market Hours | lstm | LSTM | 598 | 257 | 341 | 42.98% | 42.50% | 42.92% | 7.02 pp | -84 | 55 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 598 | 255 | 343 | 42.64% | 45.00% | 43.54% | 7.36 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 598 | 254 | 344 | 42.47% | 42.08% | 42.08% | 7.53 pp | -90 | 55 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 652 | 269 | 383 | 41.26% | 41.67% | 40.83% | 8.74 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 652 | 268 | 384 | 41.10% | 42.50% | 40.62% | 8.90 pp | -116 | 55 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 652 | 267 | 385 | 40.95% | 42.50% | 40.42% | 9.05 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 1004 | 443 | 561 | 44.12% | 42.50% | 41.46% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1004 | 443 | 561 | 44.12% | 42.08% | 43.54% | 5.88 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Hourly | lstm | LSTM | 1004 | 425 | 579 | 42.33% | 36.67% | 40.00% | 7.67 pp | -154 | 52 | -2.96 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| BTC Daily | rf | RandomForest | 826 | 342 | 484 | 41.40% | 37.08% | 40.42% | 8.60 pp | -142 | 47 | -3.02 |
| Consolidated Hourly | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1004 | 414 | 590 | 41.24% | 36.25% | 38.96% | 8.76 pp | -176 | 52 | -3.38 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 836 | 328 | 508 | 39.23% | 36.67% | 36.25% | 10.77 pp | -180 | 47 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1004 | 475 | 529 | 47.31% | 47.92% | 45.83% | 2.69 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1004 | 468 | 536 | 46.61% | 46.25% | 45.00% | 3.39 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1004 | 443 | 561 | 44.12% | 42.50% | 41.46% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1004 | 443 | 561 | 44.12% | 42.08% | 43.54% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | lstm | LSTM | 1004 | 425 | 579 | 42.33% | 36.67% | 40.00% | 7.67 pp | -154 | 52 | -2.96 |
| BTC Hourly | xgb | XGBoost | 1004 | 414 | 590 | 41.24% | 36.25% | 38.96% | 8.76 pp | -176 | 52 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 826 | 394 | 432 | 47.70% | 45.00% | 46.25% | 2.30 pp | -38 | 47 | -0.81 |
| BTC Daily | nn | NN | 826 | 385 | 441 | 46.61% | 45.42% | 45.42% | 3.39 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 826 | 381 | 445 | 46.13% | 37.92% | 44.79% | 3.87 pp | -64 | 47 | -1.36 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Daily | rf | RandomForest | 826 | 342 | 484 | 41.40% | 37.08% | 40.42% | 8.60 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 836 | 328 | 508 | 39.23% | 36.67% | 36.25% | 10.77 pp | -180 | 47 | -3.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 598 | 288 | 310 | 48.16% | 46.25% | 47.29% | 1.84 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 598 | 287 | 311 | 47.99% | 51.67% | 49.79% | 2.01 pp | -24 | 55 | -0.44 |
| BTC Market Hours | transformer | Transformer | 598 | 280 | 318 | 46.82% | 45.83% | 46.25% | 3.18 pp | -38 | 55 | -0.69 |
| BTC Market Hours | lstm | LSTM | 598 | 257 | 341 | 42.98% | 42.50% | 42.92% | 7.02 pp | -84 | 55 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 598 | 255 | 343 | 42.64% | 45.00% | 43.54% | 7.36 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 598 | 254 | 344 | 42.47% | 42.08% | 42.08% | 7.53 pp | -90 | 55 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 652 | 305 | 347 | 46.78% | 48.33% | 47.50% | 3.22 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 652 | 304 | 348 | 46.63% | 47.92% | 47.08% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 652 | 304 | 348 | 46.63% | 47.92% | 47.92% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 652 | 269 | 383 | 41.26% | 41.67% | 40.83% | 8.74 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 652 | 268 | 384 | 41.10% | 42.50% | 40.62% | 8.90 pp | -116 | 55 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 652 | 267 | 385 | 40.95% | 42.50% | 40.42% | 9.05 pp | -118 | 55 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
