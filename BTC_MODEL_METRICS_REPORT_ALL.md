# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T11:10:45.715381+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1339 | 1051 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1215 | 850 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 974 | 612 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 975 | 665 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 250 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 23:00:00+00:00 | 250 | 91 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 612 | 295 | 317 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 612 | 293 | 319 | 47.88% | 50.83% | 49.79% | 2.12 pp | -26 | 56 | -0.46 |
| BTC Market Hours Daily | nn | NN | 665 | 313 | 352 | 47.07% | 49.58% | 48.12% | 2.93 pp | -39 | 56 | -0.70 |
| BTC Market Hours | transformer | Transformer | 612 | 286 | 326 | 46.73% | 46.25% | 45.62% | 3.27 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 665 | 311 | 354 | 46.77% | 48.75% | 47.08% | 3.23 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 400 | 440 | 47.62% | 44.17% | 46.04% | 2.38 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 665 | 309 | 356 | 46.47% | 48.33% | 47.92% | 3.53 pp | -47 | 56 | -0.84 |
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1017 | 482 | 535 | 47.39% | 47.92% | 46.04% | 2.61 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.42% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Hourly | transformer | Transformer | 1017 | 476 | 541 | 46.80% | 46.67% | 44.79% | 3.20 pp | -65 | 52 | -1.25 |
| Consolidated Market Hours | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.50% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| BTC Market Hours | lstm | LSTM | 612 | 262 | 350 | 42.81% | 42.92% | 42.92% | 7.19 pp | -88 | 56 | -1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| BTC Market Hours | rf | RandomForest | 612 | 261 | 351 | 42.65% | 42.92% | 41.88% | 7.35 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 612 | 260 | 352 | 42.48% | 46.25% | 43.12% | 7.52 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 665 | 276 | 389 | 41.50% | 43.33% | 41.67% | 8.50 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 665 | 272 | 393 | 40.90% | 43.75% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 665 | 272 | 393 | 40.90% | 43.33% | 40.83% | 9.10 pp | -121 | 56 | -2.16 |
| BTC Hourly | nn | NN | 1017 | 448 | 569 | 44.05% | 41.25% | 40.83% | 5.95 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1017 | 445 | 572 | 43.76% | 40.42% | 42.08% | 6.24 pp | -127 | 52 | -2.44 |
| Consolidated Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| BTC Daily | lstm | LSTM | 840 | 354 | 486 | 42.14% | 35.83% | 39.58% | 7.86 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.25% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |
| BTC Hourly | lstm | LSTM | 1017 | 428 | 589 | 42.08% | 35.00% | 39.17% | 7.92 pp | -161 | 52 | -3.10 |
| BTC Hourly | xgb | XGBoost | 1017 | 416 | 601 | 40.90% | 34.58% | 37.50% | 9.10 pp | -185 | 52 | -3.56 |
| Consolidated Market Hours | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.50% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1017 | 482 | 535 | 47.39% | 47.92% | 46.04% | 2.61 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1017 | 476 | 541 | 46.80% | 46.67% | 44.79% | 3.20 pp | -65 | 52 | -1.25 |
| BTC Hourly | nn | NN | 1017 | 448 | 569 | 44.05% | 41.25% | 40.83% | 5.95 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1017 | 445 | 572 | 43.76% | 40.42% | 42.08% | 6.24 pp | -127 | 52 | -2.44 |
| BTC Hourly | lstm | LSTM | 1017 | 428 | 589 | 42.08% | 35.00% | 39.17% | 7.92 pp | -161 | 52 | -3.10 |
| BTC Hourly | xgb | XGBoost | 1017 | 416 | 601 | 40.90% | 34.58% | 37.50% | 9.10 pp | -185 | 52 | -3.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 400 | 440 | 47.62% | 44.17% | 46.04% | 2.38 pp | -40 | 48 | -0.83 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.42% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.50% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 840 | 354 | 486 | 42.14% | 35.83% | 39.58% | 7.86 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.25% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.50% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 612 | 295 | 317 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 612 | 293 | 319 | 47.88% | 50.83% | 49.79% | 2.12 pp | -26 | 56 | -0.46 |
| BTC Market Hours | transformer | Transformer | 612 | 286 | 326 | 46.73% | 46.25% | 45.62% | 3.27 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 612 | 262 | 350 | 42.81% | 42.92% | 42.92% | 7.19 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 612 | 261 | 351 | 42.65% | 42.92% | 41.88% | 7.35 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 612 | 260 | 352 | 42.48% | 46.25% | 43.12% | 7.52 pp | -92 | 56 | -1.64 |

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
| Consolidated Hourly | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Hourly | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 250 | 118 | 132 | 47.20% | 47.08% | 47.20% | 2.80 pp | -14 | 15 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 250 | 115 | 135 | 46.00% | 45.00% | 46.00% | 4.00 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 250 | 113 | 137 | 45.20% | 45.42% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 250 | 113 | 137 | 45.20% | 44.58% | 45.20% | 4.80 pp | -24 | 15 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 250 | 103 | 147 | 41.20% | 40.83% | 41.20% | 8.80 pp | -44 | 15 | -2.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 250 | 102 | 148 | 40.80% | 41.25% | 40.80% | 9.20 pp | -46 | 15 | -3.07 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
