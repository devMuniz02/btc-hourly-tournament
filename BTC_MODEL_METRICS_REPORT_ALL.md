# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T12:25:51.354029+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1340 | 1052 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1215 | 850 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 974 | 612 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 976 | 666 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 612 | 295 | 317 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 612 | 293 | 319 | 47.88% | 50.83% | 49.79% | 2.12 pp | -26 | 56 | -0.46 |
| BTC Market Hours Daily | nn | NN | 666 | 314 | 352 | 47.15% | 50.00% | 48.33% | 2.85 pp | -38 | 56 | -0.68 |
| BTC Market Hours | transformer | Transformer | 612 | 286 | 326 | 46.73% | 46.25% | 45.62% | 3.27 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 666 | 312 | 354 | 46.85% | 49.17% | 47.29% | 3.15 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 666 | 310 | 356 | 46.55% | 48.33% | 48.12% | 3.45 pp | -46 | 56 | -0.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 399 | 441 | 47.50% | 44.17% | 45.83% | 2.50 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1018 | 482 | 536 | 47.35% | 47.50% | 46.04% | 2.65 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.83% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Hourly | transformer | Transformer | 1018 | 477 | 541 | 46.86% | 46.67% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.92% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Market Hours | lstm | LSTM | 612 | 262 | 350 | 42.81% | 42.92% | 42.92% | 7.19 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 612 | 261 | 351 | 42.65% | 42.92% | 41.88% | 7.35 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 612 | 260 | 352 | 42.48% | 46.25% | 43.12% | 7.52 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 666 | 276 | 390 | 41.44% | 43.33% | 41.67% | 8.56 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 666 | 273 | 393 | 40.99% | 44.17% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 666 | 273 | 393 | 40.99% | 43.75% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1018 | 448 | 570 | 44.01% | 41.25% | 40.83% | 5.99 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1018 | 446 | 572 | 43.81% | 40.83% | 42.08% | 6.19 pp | -126 | 52 | -2.42 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| BTC Daily | lstm | LSTM | 840 | 355 | 485 | 42.26% | 36.25% | 39.79% | 7.74 pp | -130 | 48 | -2.71 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.67% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1018 | 428 | 590 | 42.04% | 35.00% | 39.17% | 7.96 pp | -162 | 52 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| BTC Hourly | xgb | XGBoost | 1018 | 417 | 601 | 40.96% | 35.00% | 37.50% | 9.04 pp | -184 | 52 | -3.54 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.92% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1018 | 482 | 536 | 47.35% | 47.50% | 46.04% | 2.65 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1018 | 477 | 541 | 46.86% | 46.67% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| BTC Hourly | nn | NN | 1018 | 448 | 570 | 44.01% | 41.25% | 40.83% | 5.99 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1018 | 446 | 572 | 43.81% | 40.83% | 42.08% | 6.19 pp | -126 | 52 | -2.42 |
| BTC Hourly | lstm | LSTM | 1018 | 428 | 590 | 42.04% | 35.00% | 39.17% | 7.96 pp | -162 | 52 | -3.12 |
| BTC Hourly | xgb | XGBoost | 1018 | 417 | 601 | 40.96% | 35.00% | 37.50% | 9.04 pp | -184 | 52 | -3.54 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 399 | 441 | 47.50% | 44.17% | 45.83% | 2.50 pp | -42 | 48 | -0.88 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.83% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.92% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 840 | 355 | 485 | 42.26% | 36.25% | 39.79% | 7.74 pp | -130 | 48 | -2.71 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.67% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.92% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

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
| BTC Market Hours Daily | nn | NN | 666 | 314 | 352 | 47.15% | 50.00% | 48.33% | 2.85 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 666 | 312 | 354 | 46.85% | 49.17% | 47.29% | 3.15 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 666 | 310 | 356 | 46.55% | 48.33% | 48.12% | 3.45 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 666 | 276 | 390 | 41.44% | 43.33% | 41.67% | 8.56 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 666 | 273 | 393 | 40.99% | 44.17% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 666 | 273 | 393 | 40.99% | 43.75% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
