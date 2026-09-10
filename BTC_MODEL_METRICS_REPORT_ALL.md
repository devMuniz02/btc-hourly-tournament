# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T13:54:14.959445+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 948 | 598 | 349 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 950 | 652 | 296 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 598 | 288 | 310 | 48.16% | 46.25% | 47.29% | 1.84 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 598 | 287 | 311 | 47.99% | 51.67% | 49.79% | 2.01 pp | -24 | 55 | -0.44 |
| BTC Market Hours | transformer | Transformer | 598 | 280 | 318 | 46.82% | 45.83% | 46.25% | 3.18 pp | -38 | 55 | -0.69 |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | nn | NN | 652 | 305 | 347 | 46.78% | 48.33% | 47.50% | 3.22 pp | -42 | 55 | -0.76 |
| BTC Daily | mlp_sklearn | MLPClassifier | 826 | 395 | 431 | 47.82% | 45.00% | 46.46% | 2.18 pp | -36 | 47 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 652 | 304 | 348 | 46.63% | 47.92% | 47.08% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 652 | 304 | 348 | 46.63% | 47.92% | 47.92% | 3.37 pp | -44 | 55 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1003 | 475 | 528 | 47.36% | 48.33% | 46.04% | 2.64 pp | -53 | 52 | -1.02 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 826 | 386 | 440 | 46.73% | 45.42% | 45.62% | 3.27 pp | -54 | 47 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Daily | transformer | Transformer | 826 | 382 | 444 | 46.25% | 37.92% | 45.00% | 3.75 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1003 | 467 | 536 | 46.56% | 46.25% | 45.00% | 3.44 pp | -69 | 52 | -1.33 |
| BTC Market Hours | lstm | LSTM | 598 | 257 | 341 | 42.98% | 42.50% | 42.92% | 7.02 pp | -84 | 55 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 598 | 255 | 343 | 42.64% | 45.00% | 43.54% | 7.36 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 598 | 254 | 344 | 42.47% | 42.08% | 42.08% | 7.53 pp | -90 | 55 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 652 | 269 | 383 | 41.26% | 41.67% | 40.83% | 8.74 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 652 | 268 | 384 | 41.10% | 42.50% | 40.62% | 8.90 pp | -116 | 55 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 652 | 267 | 385 | 40.95% | 42.50% | 40.42% | 9.05 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 1003 | 443 | 560 | 44.17% | 42.50% | 41.67% | 5.83 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1003 | 443 | 560 | 44.17% | 42.50% | 43.75% | 5.83 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Hourly | lstm | LSTM | 1003 | 425 | 578 | 42.37% | 36.67% | 40.21% | 7.63 pp | -153 | 52 | -2.94 |
| BTC Daily | rf | RandomForest | 826 | 343 | 483 | 41.53% | 37.50% | 40.62% | 8.47 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| BTC Hourly | xgb | XGBoost | 1003 | 414 | 589 | 41.28% | 36.25% | 39.17% | 8.72 pp | -175 | 52 | -3.37 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 836 | 329 | 507 | 39.35% | 36.67% | 36.46% | 10.65 pp | -178 | 47 | -3.79 |

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
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
