# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T14:25:45.796365+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 1 | 0 | 0 |

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
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1003 | 475 | 528 | 47.36% | 48.33% | 46.04% | 2.64 pp | -53 | 52 | -1.02 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 826 | 386 | 440 | 46.73% | 45.42% | 45.62% | 3.27 pp | -54 | 47 | -1.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 826 | 382 | 444 | 46.25% | 37.92% | 45.00% | 3.75 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1003 | 467 | 536 | 46.56% | 46.25% | 45.00% | 3.44 pp | -69 | 52 | -1.33 |
| BTC Market Hours | lstm | LSTM | 598 | 257 | 341 | 42.98% | 42.50% | 42.92% | 7.02 pp | -84 | 55 | -1.53 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 598 | 255 | 343 | 42.64% | 45.00% | 43.54% | 7.36 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 598 | 254 | 344 | 42.47% | 42.08% | 42.08% | 7.53 pp | -90 | 55 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 652 | 269 | 383 | 41.26% | 41.67% | 40.83% | 8.74 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 652 | 268 | 384 | 41.10% | 42.50% | 40.62% | 8.90 pp | -116 | 55 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 652 | 267 | 385 | 40.95% | 42.50% | 40.42% | 9.05 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 1003 | 443 | 560 | 44.17% | 42.50% | 41.67% | 5.83 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1003 | 443 | 560 | 44.17% | 42.50% | 43.75% | 5.83 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 34 | 52 | 39.53% | 39.53% | 39.53% | 10.47 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 826 | 348 | 478 | 42.13% | 35.00% | 40.21% | 7.87 pp | -130 | 47 | -2.77 |
| BTC Hourly | lstm | LSTM | 1003 | 425 | 578 | 42.37% | 36.67% | 40.21% | 7.63 pp | -153 | 52 | -2.94 |
| BTC Daily | rf | RandomForest | 826 | 343 | 483 | 41.53% | 37.50% | 40.62% | 8.47 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1003 | 414 | 589 | 41.28% | 36.25% | 39.17% | 8.72 pp | -175 | 52 | -3.37 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 86 | 30 | 56 | 34.88% | 34.88% | 34.88% | 15.12 pp | -26 | 7 | -3.71 |
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
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 34 | 52 | 39.53% | 39.53% | 39.53% | 10.47 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 86 | 30 | 56 | 34.88% | 34.88% | 34.88% | 15.12 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
