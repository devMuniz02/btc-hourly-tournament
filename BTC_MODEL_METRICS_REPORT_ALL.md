# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T14:04:04.815683+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1325 | 1037 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1201 | 836 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 13:00:00+00:00 | 949 | 598 | 350 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 13:00:00+00:00 | 951 | 652 | 297 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 238 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 238 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 238 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T16:00:00+00:00 | 239 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 598 | 288 | 310 | 48.16% | 46.25% | 47.29% | 1.84 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 598 | 287 | 311 | 47.99% | 51.67% | 49.79% | 2.01 pp | -24 | 55 | -0.44 |
| BTC Market Hours | transformer | Transformer | 598 | 280 | 318 | 46.82% | 45.83% | 46.25% | 3.18 pp | -38 | 55 | -0.69 |
| BTC Market Hours Daily | nn | NN | 652 | 305 | 347 | 46.78% | 48.33% | 47.50% | 3.22 pp | -42 | 55 | -0.76 |
| BTC Daily | mlp_sklearn | MLPClassifier | 826 | 395 | 431 | 47.82% | 45.00% | 46.46% | 2.18 pp | -36 | 47 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 652 | 304 | 348 | 46.63% | 47.92% | 47.08% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 652 | 304 | 348 | 46.63% | 47.92% | 47.92% | 3.37 pp | -44 | 55 | -0.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 7 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1003 | 475 | 528 | 47.36% | 48.33% | 46.04% | 2.64 pp | -53 | 52 | -1.02 |
| Consolidated Hourly | rf | RandomForest | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 15 | -1.07 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 826 | 386 | 440 | 46.73% | 45.42% | 45.62% | 3.27 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 826 | 382 | 444 | 46.25% | 37.92% | 45.00% | 3.75 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1003 | 467 | 536 | 46.56% | 46.25% | 45.00% | 3.44 pp | -69 | 52 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 238 | 109 | 129 | 45.80% | 45.80% | 45.80% | 4.20 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 238 | 109 | 129 | 45.80% | 45.80% | 45.80% | 4.20 pp | -20 | 15 | -1.33 |
| BTC Market Hours | lstm | LSTM | 598 | 257 | 341 | 42.98% | 42.50% | 42.92% | 7.02 pp | -84 | 55 | -1.53 |
| Consolidated Hourly | lstm | LSTM | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 15 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 598 | 255 | 343 | 42.64% | 45.00% | 43.54% | 7.36 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 598 | 254 | 344 | 42.47% | 42.08% | 42.08% | 7.53 pp | -90 | 55 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 15 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 652 | 269 | 383 | 41.26% | 41.67% | 40.83% | 8.74 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 652 | 268 | 384 | 41.10% | 42.50% | 40.62% | 8.90 pp | -116 | 55 | -2.11 |
| Consolidated Hourly | xgb | XGBoost | 238 | 103 | 135 | 43.28% | 43.28% | 43.28% | 6.72 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 238 | 103 | 135 | 43.28% | 43.28% | 43.28% | 6.72 pp | -32 | 15 | -2.13 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 652 | 267 | 385 | 40.95% | 42.50% | 40.42% | 9.05 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 1003 | 443 | 560 | 44.17% | 42.50% | 41.67% | 5.83 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1003 | 443 | 560 | 44.17% | 42.50% | 43.75% | 5.83 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 238 | 99 | 139 | 41.60% | 41.60% | 41.60% | 8.40 pp | -40 | 15 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 238 | 99 | 139 | 41.60% | 41.60% | 41.60% | 8.40 pp | -40 | 15 | -2.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Hourly | lstm | LSTM | 1003 | 425 | 578 | 42.37% | 36.67% | 40.21% | 7.63 pp | -153 | 52 | -2.94 |
| BTC Daily | rf | RandomForest | 826 | 343 | 483 | 41.53% | 37.50% | 40.62% | 8.47 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1003 | 414 | 589 | 41.28% | 36.25% | 39.17% | 8.72 pp | -175 | 52 | -3.37 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 836 | 329 | 507 | 39.35% | 36.67% | 36.46% | 10.65 pp | -178 | 47 | -3.79 |
| Consolidated Market Hours Daily | nn | NN | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 7 | -3.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1003 | 475 | 528 | 47.36% | 48.33% | 46.04% | 2.64 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1003 | 467 | 536 | 46.56% | 46.25% | 45.00% | 3.44 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1003 | 443 | 560 | 44.17% | 42.50% | 41.67% | 5.83 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1003 | 443 | 560 | 44.17% | 42.50% | 43.75% | 5.83 pp | -117 | 52 | -2.25 |
| BTC Hourly | lstm | LSTM | 1003 | 425 | 578 | 42.37% | 36.67% | 40.21% | 7.63 pp | -153 | 52 | -2.94 |
| BTC Hourly | xgb | XGBoost | 1003 | 414 | 589 | 41.28% | 36.25% | 39.17% | 8.72 pp | -175 | 52 | -3.37 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 826 | 395 | 431 | 47.82% | 45.00% | 46.46% | 2.18 pp | -36 | 47 | -0.77 |
| BTC Daily | nn | NN | 826 | 386 | 440 | 46.73% | 45.42% | 45.62% | 3.27 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 826 | 382 | 444 | 46.25% | 37.92% | 45.00% | 3.75 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Daily | rf | RandomForest | 826 | 343 | 483 | 41.53% | 37.50% | 40.62% | 8.47 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 836 | 329 | 507 | 39.35% | 36.67% | 36.46% | 10.65 pp | -178 | 47 | -3.79 |

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
| Consolidated Hourly | rf | RandomForest | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 238 | 109 | 129 | 45.80% | 45.80% | 45.80% | 4.20 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 238 | 103 | 135 | 43.28% | 43.28% | 43.28% | 6.72 pp | -32 | 15 | -2.13 |
| Consolidated Hourly | nn | NN | 238 | 99 | 139 | 41.60% | 41.60% | 41.60% | 8.40 pp | -40 | 15 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 238 | 109 | 129 | 45.80% | 45.80% | 45.80% | 4.20 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 238 | 103 | 135 | 43.28% | 43.28% | 43.28% | 6.72 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 238 | 99 | 139 | 41.60% | 41.60% | 41.60% | 8.40 pp | -40 | 15 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 7 | -3.86 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
