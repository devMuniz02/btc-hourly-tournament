# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T07:31:42.212041+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1289 | 1001 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1165 | 800 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 885 | 562 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 887 | 616 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T01:00:00+00:00 | 204 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T01:00:00+00:00 | 204 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T01:00:00+00:00 | 204 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T01:00:00+00:00 | 205 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.08% | 47.92% | 1.25 pp | -14 | 53 | -0.26 |
| Consolidated Hourly | rf | RandomForest | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 14 | -0.29 |
| BTC Market Hours | nn | NN | 562 | 267 | 295 | 47.51% | 51.25% | 49.17% | 2.49 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 382 | 408 | 48.35% | 47.50% | 48.33% | 1.65 pp | -26 | 46 | -0.57 |
| BTC Market Hours | transformer | Transformer | 562 | 264 | 298 | 46.98% | 46.25% | 47.29% | 3.02 pp | -34 | 53 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 14 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 616 | 288 | 328 | 46.75% | 49.17% | 47.71% | 3.25 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 616 | 287 | 329 | 46.59% | 49.17% | 47.29% | 3.41 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | nn | NN | 616 | 287 | 329 | 46.59% | 47.50% | 48.12% | 3.41 pp | -42 | 53 | -0.79 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 967 | 460 | 507 | 47.57% | 49.17% | 46.46% | 2.43 pp | -47 | 50 | -0.94 |
| Consolidated Hourly | xgb | XGBoost | 204 | 94 | 110 | 46.08% | 46.08% | 46.08% | 3.92 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 204 | 94 | 110 | 46.08% | 46.08% | 46.08% | 3.92 pp | -16 | 14 | -1.14 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 967 | 450 | 517 | 46.54% | 44.17% | 43.75% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 562 | 245 | 317 | 43.59% | 42.50% | 44.17% | 6.41 pp | -72 | 53 | -1.36 |
| Consolidated Hourly | lstm | LSTM | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| BTC Market Hours | rf | RandomForest | 562 | 243 | 319 | 43.24% | 45.83% | 43.54% | 6.76 pp | -76 | 53 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 562 | 239 | 323 | 42.53% | 45.83% | 42.92% | 7.47 pp | -84 | 53 | -1.58 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 14 | -1.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 616 | 258 | 358 | 41.88% | 44.58% | 41.04% | 8.12 pp | -100 | 53 | -1.89 |
| BTC Market Hours Daily | xgb | XGBoost | 616 | 252 | 364 | 40.91% | 43.75% | 40.62% | 9.09 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 616 | 251 | 365 | 40.75% | 40.83% | 40.42% | 9.25 pp | -114 | 53 | -2.15 |
| BTC Hourly | rf | RandomForest | 967 | 428 | 539 | 44.26% | 42.08% | 42.50% | 5.74 pp | -111 | 50 | -2.22 |
| BTC Hourly | nn | NN | 967 | 427 | 540 | 44.16% | 41.25% | 42.50% | 5.84 pp | -113 | 50 | -2.26 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 331 | 459 | 41.90% | 38.75% | 41.67% | 8.10 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 967 | 412 | 555 | 42.61% | 37.50% | 41.46% | 7.39 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 967 | 399 | 568 | 41.26% | 36.67% | 38.75% | 8.74 pp | -169 | 50 | -3.38 |
| BTC Daily | xgb | XGBoost | 800 | 313 | 487 | 39.12% | 36.25% | 36.25% | 10.88 pp | -174 | 46 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 967 | 460 | 507 | 47.57% | 49.17% | 46.46% | 2.43 pp | -47 | 50 | -0.94 |
| BTC Hourly | transformer | Transformer | 967 | 450 | 517 | 46.54% | 44.17% | 43.75% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Hourly | rf | RandomForest | 967 | 428 | 539 | 44.26% | 42.08% | 42.50% | 5.74 pp | -111 | 50 | -2.22 |
| BTC Hourly | nn | NN | 967 | 427 | 540 | 44.16% | 41.25% | 42.50% | 5.84 pp | -113 | 50 | -2.26 |
| BTC Hourly | lstm | LSTM | 967 | 412 | 555 | 42.61% | 37.50% | 41.46% | 7.39 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 967 | 399 | 568 | 41.26% | 36.67% | 38.75% | 8.74 pp | -169 | 50 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 382 | 408 | 48.35% | 47.50% | 48.33% | 1.65 pp | -26 | 46 | -0.57 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 331 | 459 | 41.90% | 38.75% | 41.67% | 8.10 pp | -128 | 46 | -2.78 |
| BTC Daily | xgb | XGBoost | 800 | 313 | 487 | 39.12% | 36.25% | 36.25% | 10.88 pp | -174 | 46 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.08% | 47.92% | 1.25 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 562 | 267 | 295 | 47.51% | 51.25% | 49.17% | 2.49 pp | -28 | 53 | -0.53 |
| BTC Market Hours | transformer | Transformer | 562 | 264 | 298 | 46.98% | 46.25% | 47.29% | 3.02 pp | -34 | 53 | -0.64 |
| BTC Market Hours | lstm | LSTM | 562 | 245 | 317 | 43.59% | 42.50% | 44.17% | 6.41 pp | -72 | 53 | -1.36 |
| BTC Market Hours | rf | RandomForest | 562 | 243 | 319 | 43.24% | 45.83% | 43.54% | 6.76 pp | -76 | 53 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 562 | 239 | 323 | 42.53% | 45.83% | 42.92% | 7.47 pp | -84 | 53 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 616 | 288 | 328 | 46.75% | 49.17% | 47.71% | 3.25 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 616 | 287 | 329 | 46.59% | 49.17% | 47.29% | 3.41 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | nn | NN | 616 | 287 | 329 | 46.59% | 47.50% | 48.12% | 3.41 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 616 | 258 | 358 | 41.88% | 44.58% | 41.04% | 8.12 pp | -100 | 53 | -1.89 |
| BTC Market Hours Daily | xgb | XGBoost | 616 | 252 | 364 | 40.91% | 43.75% | 40.62% | 9.09 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 616 | 251 | 365 | 40.75% | 40.83% | 40.42% | 9.25 pp | -114 | 53 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 14 | -0.71 |
| Consolidated Hourly | xgb | XGBoost | 204 | 94 | 110 | 46.08% | 46.08% | 46.08% | 3.92 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 14 | -1.86 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 204 | 94 | 110 | 46.08% | 46.08% | 46.08% | 3.92 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 14 | -1.86 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
