# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T21:45:27.925862+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1266 | 978 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1142 | 777 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 845 | 539 | 305 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 847 | 593 | 252 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 184 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 539 | 262 | 277 | 48.61% | 46.25% | 48.33% | 1.39 pp | -15 | 51 | -0.29 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| BTC Market Hours | transformer | Transformer | 539 | 258 | 281 | 47.87% | 49.17% | 48.33% | 2.13 pp | -23 | 51 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 767 | 372 | 395 | 48.50% | 47.50% | 48.75% | 1.50 pp | -23 | 45 | -0.51 |
| BTC Market Hours Daily | transformer | Transformer | 593 | 282 | 311 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours | nn | NN | 539 | 255 | 284 | 47.31% | 50.42% | 48.96% | 2.69 pp | -29 | 51 | -0.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | nn | NN | 593 | 276 | 317 | 46.54% | 46.25% | 48.12% | 3.46 pp | -41 | 51 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 593 | 275 | 318 | 46.37% | 50.83% | 47.29% | 3.63 pp | -43 | 51 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 944 | 451 | 493 | 47.78% | 50.42% | 47.08% | 2.22 pp | -42 | 49 | -0.86 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 767 | 360 | 407 | 46.94% | 42.08% | 47.29% | 3.06 pp | -47 | 45 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| BTC Daily | nn | NN | 767 | 356 | 411 | 46.41% | 45.00% | 46.25% | 3.59 pp | -55 | 45 | -1.22 |
| BTC Hourly | transformer | Transformer | 944 | 442 | 502 | 46.82% | 46.25% | 45.00% | 3.18 pp | -60 | 49 | -1.22 |
| BTC Market Hours | rf | RandomForest | 539 | 233 | 306 | 43.23% | 45.42% | 43.33% | 6.77 pp | -73 | 51 | -1.43 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| BTC Market Hours | lstm | LSTM | 539 | 232 | 307 | 43.04% | 42.08% | 43.96% | 6.96 pp | -75 | 51 | -1.47 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 539 | 224 | 315 | 41.56% | 43.33% | 41.88% | 8.44 pp | -91 | 51 | -1.78 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 593 | 248 | 345 | 41.82% | 45.42% | 41.67% | 8.18 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Hourly | nn | NN | 944 | 420 | 524 | 44.49% | 42.92% | 42.92% | 5.51 pp | -104 | 49 | -2.12 |
| BTC Hourly | rf | RandomForest | 944 | 419 | 525 | 44.39% | 44.17% | 43.75% | 5.61 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 593 | 240 | 353 | 40.47% | 39.17% | 40.00% | 9.53 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 593 | 236 | 357 | 39.80% | 41.25% | 38.96% | 10.20 pp | -121 | 51 | -2.37 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 767 | 325 | 442 | 42.37% | 36.25% | 40.62% | 7.63 pp | -117 | 45 | -2.60 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 767 | 322 | 445 | 41.98% | 38.75% | 42.50% | 8.02 pp | -123 | 45 | -2.73 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | lstm | LSTM | 944 | 403 | 541 | 42.69% | 36.25% | 42.08% | 7.31 pp | -138 | 49 | -2.82 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 944 | 396 | 548 | 41.95% | 40.42% | 40.62% | 8.05 pp | -152 | 49 | -3.10 |
| BTC Daily | xgb | XGBoost | 777 | 305 | 472 | 39.25% | 35.83% | 36.67% | 10.75 pp | -167 | 45 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 944 | 451 | 493 | 47.78% | 50.42% | 47.08% | 2.22 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 944 | 442 | 502 | 46.82% | 46.25% | 45.00% | 3.18 pp | -60 | 49 | -1.22 |
| BTC Hourly | nn | NN | 944 | 420 | 524 | 44.49% | 42.92% | 42.92% | 5.51 pp | -104 | 49 | -2.12 |
| BTC Hourly | rf | RandomForest | 944 | 419 | 525 | 44.39% | 44.17% | 43.75% | 5.61 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 944 | 403 | 541 | 42.69% | 36.25% | 42.08% | 7.31 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 944 | 396 | 548 | 41.95% | 40.42% | 40.62% | 8.05 pp | -152 | 49 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 767 | 372 | 395 | 48.50% | 47.50% | 48.75% | 1.50 pp | -23 | 45 | -0.51 |
| BTC Daily | transformer | Transformer | 767 | 360 | 407 | 46.94% | 42.08% | 47.29% | 3.06 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 767 | 356 | 411 | 46.41% | 45.00% | 46.25% | 3.59 pp | -55 | 45 | -1.22 |
| BTC Daily | lstm | LSTM | 767 | 325 | 442 | 42.37% | 36.25% | 40.62% | 7.63 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 767 | 322 | 445 | 41.98% | 38.75% | 42.50% | 8.02 pp | -123 | 45 | -2.73 |
| BTC Daily | xgb | XGBoost | 777 | 305 | 472 | 39.25% | 35.83% | 36.67% | 10.75 pp | -167 | 45 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 539 | 262 | 277 | 48.61% | 46.25% | 48.33% | 1.39 pp | -15 | 51 | -0.29 |
| BTC Market Hours | transformer | Transformer | 539 | 258 | 281 | 47.87% | 49.17% | 48.33% | 2.13 pp | -23 | 51 | -0.45 |
| BTC Market Hours | nn | NN | 539 | 255 | 284 | 47.31% | 50.42% | 48.96% | 2.69 pp | -29 | 51 | -0.57 |
| BTC Market Hours | rf | RandomForest | 539 | 233 | 306 | 43.23% | 45.42% | 43.33% | 6.77 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 539 | 232 | 307 | 43.04% | 42.08% | 43.96% | 6.96 pp | -75 | 51 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 539 | 224 | 315 | 41.56% | 43.33% | 41.88% | 8.44 pp | -91 | 51 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 593 | 282 | 311 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours Daily | nn | NN | 593 | 276 | 317 | 46.54% | 46.25% | 48.12% | 3.46 pp | -41 | 51 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 593 | 275 | 318 | 46.37% | 50.83% | 47.29% | 3.63 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 593 | 248 | 345 | 41.82% | 45.42% | 41.67% | 8.18 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 593 | 240 | 353 | 40.47% | 39.17% | 40.00% | 9.53 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 593 | 236 | 357 | 39.80% | 41.25% | 38.96% | 10.20 pp | -121 | 51 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
