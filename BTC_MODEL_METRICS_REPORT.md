# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T10:41:35.223276+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1338 | 1050 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1214 | 849 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 973 | 611 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 975 | 665 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 250 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T23:00:00+00:00 | 251 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 611 | 295 | 316 | 48.28% | 47.08% | 47.71% | 1.72 pp | -21 | 56 | -0.38 |
| BTC Market Hours | nn | NN | 611 | 292 | 319 | 47.79% | 50.83% | 49.58% | 2.21 pp | -27 | 56 | -0.48 |
| BTC Market Hours Daily | nn | NN | 665 | 313 | 352 | 47.07% | 49.58% | 48.12% | 2.93 pp | -39 | 56 | -0.70 |
| BTC Market Hours | transformer | Transformer | 611 | 286 | 325 | 46.81% | 46.67% | 45.62% | 3.19 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 665 | 311 | 354 | 46.77% | 48.75% | 47.08% | 3.23 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 839 | 400 | 439 | 47.68% | 44.58% | 46.04% | 2.32 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 665 | 309 | 356 | 46.47% | 48.33% | 47.92% | 3.53 pp | -47 | 56 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1016 | 481 | 535 | 47.34% | 47.92% | 46.04% | 2.66 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 839 | 391 | 448 | 46.60% | 45.00% | 45.21% | 3.40 pp | -57 | 48 | -1.19 |
| Consolidated Hourly | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| BTC Hourly | transformer | Transformer | 1016 | 475 | 541 | 46.75% | 46.25% | 44.79% | 3.25 pp | -66 | 52 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 839 | 386 | 453 | 46.01% | 37.50% | 44.17% | 3.99 pp | -67 | 48 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 611 | 262 | 349 | 42.88% | 43.33% | 43.12% | 7.12 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 611 | 261 | 350 | 42.72% | 43.33% | 41.88% | 7.28 pp | -89 | 56 | -1.59 |
| Consolidated Hourly | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 611 | 260 | 351 | 42.55% | 46.25% | 43.33% | 7.45 pp | -91 | 56 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 665 | 276 | 389 | 41.50% | 43.33% | 41.67% | 8.50 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 665 | 272 | 393 | 40.90% | 43.75% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 665 | 272 | 393 | 40.90% | 43.33% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |
| BTC Hourly | nn | NN | 1016 | 447 | 569 | 44.00% | 41.25% | 40.62% | 6.00 pp | -122 | 52 | -2.35 |
| Consolidated Hourly | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| BTC Hourly | rf | RandomForest | 1016 | 445 | 571 | 43.80% | 40.42% | 42.08% | 6.20 pp | -126 | 52 | -2.42 |
| Consolidated Hourly | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| BTC Daily | lstm | LSTM | 839 | 353 | 486 | 42.07% | 35.42% | 39.58% | 7.93 pp | -133 | 48 | -2.77 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 839 | 348 | 491 | 41.48% | 36.67% | 40.83% | 8.52 pp | -143 | 48 | -2.98 |
| BTC Hourly | lstm | LSTM | 1016 | 428 | 588 | 42.13% | 35.00% | 39.38% | 7.87 pp | -160 | 52 | -3.08 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| BTC Hourly | xgb | XGBoost | 1016 | 415 | 601 | 40.85% | 34.17% | 37.50% | 9.15 pp | -186 | 52 | -3.58 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 849 | 335 | 514 | 39.46% | 37.50% | 36.67% | 10.54 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1016 | 481 | 535 | 47.34% | 47.92% | 46.04% | 2.66 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1016 | 475 | 541 | 46.75% | 46.25% | 44.79% | 3.25 pp | -66 | 52 | -1.27 |
| BTC Hourly | nn | NN | 1016 | 447 | 569 | 44.00% | 41.25% | 40.62% | 6.00 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1016 | 445 | 571 | 43.80% | 40.42% | 42.08% | 6.20 pp | -126 | 52 | -2.42 |
| BTC Hourly | lstm | LSTM | 1016 | 428 | 588 | 42.13% | 35.00% | 39.38% | 7.87 pp | -160 | 52 | -3.08 |
| BTC Hourly | xgb | XGBoost | 1016 | 415 | 601 | 40.85% | 34.17% | 37.50% | 9.15 pp | -186 | 52 | -3.58 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 839 | 400 | 439 | 47.68% | 44.58% | 46.04% | 2.32 pp | -39 | 48 | -0.81 |
| BTC Daily | nn | NN | 839 | 391 | 448 | 46.60% | 45.00% | 45.21% | 3.40 pp | -57 | 48 | -1.19 |
| BTC Daily | transformer | Transformer | 839 | 386 | 453 | 46.01% | 37.50% | 44.17% | 3.99 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 839 | 353 | 486 | 42.07% | 35.42% | 39.58% | 7.93 pp | -133 | 48 | -2.77 |
| BTC Daily | rf | RandomForest | 839 | 348 | 491 | 41.48% | 36.67% | 40.83% | 8.52 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 849 | 335 | 514 | 39.46% | 37.50% | 36.67% | 10.54 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 611 | 295 | 316 | 48.28% | 47.08% | 47.71% | 1.72 pp | -21 | 56 | -0.38 |
| BTC Market Hours | nn | NN | 611 | 292 | 319 | 47.79% | 50.83% | 49.58% | 2.21 pp | -27 | 56 | -0.48 |
| BTC Market Hours | transformer | Transformer | 611 | 286 | 325 | 46.81% | 46.67% | 45.62% | 3.19 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 611 | 262 | 349 | 42.88% | 43.33% | 43.12% | 7.12 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 611 | 261 | 350 | 42.72% | 43.33% | 41.88% | 7.28 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 611 | 260 | 351 | 42.55% | 46.25% | 43.33% | 7.45 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 665 | 313 | 352 | 47.07% | 49.58% | 48.12% | 2.93 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 665 | 311 | 354 | 46.77% | 48.75% | 47.08% | 3.23 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 665 | 309 | 356 | 46.47% | 48.33% | 47.92% | 3.53 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 665 | 276 | 389 | 41.50% | 43.33% | 41.67% | 8.50 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 665 | 272 | 393 | 40.90% | 43.75% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 665 | 272 | 393 | 40.90% | 43.33% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Hourly | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 116 | 134 | 46.40% | 46.25% | 46.40% | 3.60 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 113 | 137 | 45.20% | 44.17% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 111 | 139 | 44.40% | 43.75% | 44.40% | 5.60 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 107 | 143 | 42.80% | 42.50% | 42.80% | 7.20 pp | -36 | 15 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 106 | 144 | 42.40% | 42.92% | 42.40% | 7.60 pp | -38 | 15 | -2.53 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
