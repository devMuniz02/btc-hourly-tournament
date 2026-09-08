# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T08:34:43.181517+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1290 | 1002 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1165 | 800 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 885 | 562 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 887 | 616 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.08% | 47.92% | 1.25 pp | -14 | 53 | -0.26 |
| BTC Market Hours | nn | NN | 562 | 267 | 295 | 47.51% | 51.25% | 49.17% | 2.49 pp | -28 | 53 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 381 | 409 | 48.23% | 47.08% | 48.12% | 1.77 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 562 | 264 | 298 | 46.98% | 46.25% | 47.29% | 3.02 pp | -34 | 53 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 616 | 288 | 328 | 46.75% | 49.17% | 47.71% | 3.25 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 616 | 287 | 329 | 46.59% | 49.17% | 47.29% | 3.41 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | nn | NN | 616 | 287 | 329 | 46.59% | 47.50% | 48.12% | 3.41 pp | -42 | 53 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 968 | 460 | 508 | 47.52% | 49.17% | 46.25% | 2.48 pp | -48 | 50 | -0.96 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Hourly | transformer | Transformer | 968 | 451 | 517 | 46.59% | 44.17% | 43.75% | 3.41 pp | -66 | 50 | -1.32 |
| BTC Market Hours | lstm | LSTM | 562 | 245 | 317 | 43.59% | 42.50% | 44.17% | 6.41 pp | -72 | 53 | -1.36 |
| BTC Market Hours | rf | RandomForest | 562 | 243 | 319 | 43.24% | 45.83% | 43.54% | 6.76 pp | -76 | 53 | -1.43 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 562 | 239 | 323 | 42.53% | 45.83% | 42.92% | 7.47 pp | -84 | 53 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 616 | 258 | 358 | 41.88% | 44.58% | 41.04% | 8.12 pp | -100 | 53 | -1.89 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 616 | 252 | 364 | 40.91% | 43.75% | 40.62% | 9.09 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 616 | 251 | 365 | 40.75% | 40.83% | 40.42% | 9.25 pp | -114 | 53 | -2.15 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Hourly | rf | RandomForest | 968 | 428 | 540 | 44.21% | 41.67% | 42.50% | 5.79 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 968 | 427 | 541 | 44.11% | 41.25% | 42.50% | 5.89 pp | -114 | 50 | -2.28 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 330 | 460 | 41.77% | 38.33% | 41.46% | 8.23 pp | -130 | 46 | -2.83 |
| BTC Hourly | lstm | LSTM | 968 | 412 | 556 | 42.56% | 37.50% | 41.46% | 7.44 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 968 | 399 | 569 | 41.22% | 36.25% | 38.54% | 8.78 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 800 | 313 | 487 | 39.12% | 36.25% | 36.25% | 10.88 pp | -174 | 46 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 968 | 460 | 508 | 47.52% | 49.17% | 46.25% | 2.48 pp | -48 | 50 | -0.96 |
| BTC Hourly | transformer | Transformer | 968 | 451 | 517 | 46.59% | 44.17% | 43.75% | 3.41 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 968 | 428 | 540 | 44.21% | 41.67% | 42.50% | 5.79 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 968 | 427 | 541 | 44.11% | 41.25% | 42.50% | 5.89 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 968 | 412 | 556 | 42.56% | 37.50% | 41.46% | 7.44 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 968 | 399 | 569 | 41.22% | 36.25% | 38.54% | 8.78 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 790 | 381 | 409 | 48.23% | 47.08% | 48.12% | 1.77 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 790 | 367 | 423 | 46.46% | 45.42% | 45.21% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 790 | 367 | 423 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 790 | 333 | 457 | 42.15% | 34.17% | 40.21% | 7.85 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 790 | 330 | 460 | 41.77% | 38.33% | 41.46% | 8.23 pp | -130 | 46 | -2.83 |
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
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
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
