# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T15:30:13.619947+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1342 | 1054 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1217 | 852 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 14:00:00+00:00 | 979 | 614 | 364 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 14:00:00+00:00 | 981 | 668 | 311 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 614 | 296 | 318 | 48.21% | 47.08% | 47.50% | 1.79 pp | -22 | 57 | -0.39 |
| BTC Market Hours | nn | NN | 614 | 294 | 320 | 47.88% | 50.42% | 49.58% | 2.12 pp | -26 | 57 | -0.46 |
| BTC Market Hours Daily | nn | NN | 668 | 315 | 353 | 47.16% | 50.42% | 48.54% | 2.84 pp | -38 | 57 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 668 | 313 | 355 | 46.86% | 49.17% | 47.50% | 3.14 pp | -42 | 57 | -0.74 |
| BTC Market Hours | transformer | Transformer | 614 | 286 | 328 | 46.58% | 46.25% | 45.42% | 3.42 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 668 | 310 | 358 | 46.41% | 47.92% | 48.12% | 3.59 pp | -48 | 57 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 842 | 400 | 442 | 47.51% | 44.17% | 45.83% | 2.49 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1020 | 482 | 538 | 47.25% | 47.08% | 46.04% | 2.75 pp | -56 | 52 | -1.08 |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| BTC Daily | nn | NN | 842 | 393 | 449 | 46.67% | 46.25% | 45.42% | 3.33 pp | -56 | 48 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1020 | 478 | 542 | 46.86% | 47.08% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| BTC Daily | transformer | Transformer | 842 | 387 | 455 | 45.96% | 37.50% | 44.17% | 4.04 pp | -68 | 48 | -1.42 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 614 | 264 | 350 | 43.00% | 43.33% | 43.33% | 7.00 pp | -86 | 57 | -1.51 |
| BTC Market Hours | rf | RandomForest | 614 | 263 | 351 | 42.83% | 43.33% | 41.88% | 7.17 pp | -88 | 57 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 614 | 261 | 353 | 42.51% | 46.67% | 42.92% | 7.49 pp | -92 | 57 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 668 | 277 | 391 | 41.47% | 43.33% | 41.88% | 8.53 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |
| BTC Hourly | nn | NN | 1020 | 448 | 572 | 43.92% | 40.42% | 40.62% | 6.08 pp | -124 | 52 | -2.38 |
| BTC Hourly | rf | RandomForest | 1020 | 447 | 573 | 43.82% | 40.83% | 42.29% | 6.18 pp | -126 | 52 | -2.42 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 842 | 356 | 486 | 42.28% | 36.25% | 39.79% | 7.72 pp | -130 | 48 | -2.71 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| BTC Daily | rf | RandomForest | 842 | 349 | 493 | 41.45% | 37.08% | 40.62% | 8.55 pp | -144 | 48 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| BTC Hourly | lstm | LSTM | 1020 | 428 | 592 | 41.96% | 34.17% | 39.17% | 8.04 pp | -164 | 52 | -3.15 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1020 | 418 | 602 | 40.98% | 34.58% | 37.50% | 9.02 pp | -184 | 52 | -3.54 |
| BTC Daily | xgb | XGBoost | 852 | 336 | 516 | 39.44% | 37.92% | 36.46% | 10.56 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1020 | 482 | 538 | 47.25% | 47.08% | 46.04% | 2.75 pp | -56 | 52 | -1.08 |
| BTC Hourly | transformer | Transformer | 1020 | 478 | 542 | 46.86% | 47.08% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| BTC Hourly | nn | NN | 1020 | 448 | 572 | 43.92% | 40.42% | 40.62% | 6.08 pp | -124 | 52 | -2.38 |
| BTC Hourly | rf | RandomForest | 1020 | 447 | 573 | 43.82% | 40.83% | 42.29% | 6.18 pp | -126 | 52 | -2.42 |
| BTC Hourly | lstm | LSTM | 1020 | 428 | 592 | 41.96% | 34.17% | 39.17% | 8.04 pp | -164 | 52 | -3.15 |
| BTC Hourly | xgb | XGBoost | 1020 | 418 | 602 | 40.98% | 34.58% | 37.50% | 9.02 pp | -184 | 52 | -3.54 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 842 | 400 | 442 | 47.51% | 44.17% | 45.83% | 2.49 pp | -42 | 48 | -0.88 |
| BTC Daily | nn | NN | 842 | 393 | 449 | 46.67% | 46.25% | 45.42% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 842 | 387 | 455 | 45.96% | 37.50% | 44.17% | 4.04 pp | -68 | 48 | -1.42 |
| BTC Daily | lstm | LSTM | 842 | 356 | 486 | 42.28% | 36.25% | 39.79% | 7.72 pp | -130 | 48 | -2.71 |
| BTC Daily | rf | RandomForest | 842 | 349 | 493 | 41.45% | 37.08% | 40.62% | 8.55 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 852 | 336 | 516 | 39.44% | 37.92% | 36.46% | 10.56 pp | -180 | 48 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 614 | 296 | 318 | 48.21% | 47.08% | 47.50% | 1.79 pp | -22 | 57 | -0.39 |
| BTC Market Hours | nn | NN | 614 | 294 | 320 | 47.88% | 50.42% | 49.58% | 2.12 pp | -26 | 57 | -0.46 |
| BTC Market Hours | transformer | Transformer | 614 | 286 | 328 | 46.58% | 46.25% | 45.42% | 3.42 pp | -42 | 57 | -0.74 |
| BTC Market Hours | lstm | LSTM | 614 | 264 | 350 | 43.00% | 43.33% | 43.33% | 7.00 pp | -86 | 57 | -1.51 |
| BTC Market Hours | rf | RandomForest | 614 | 263 | 351 | 42.83% | 43.33% | 41.88% | 7.17 pp | -88 | 57 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 614 | 261 | 353 | 42.51% | 46.67% | 42.92% | 7.49 pp | -92 | 57 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 668 | 315 | 353 | 47.16% | 50.42% | 48.54% | 2.84 pp | -38 | 57 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 668 | 313 | 355 | 46.86% | 49.17% | 47.50% | 3.14 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 668 | 310 | 358 | 46.41% | 47.92% | 48.12% | 3.59 pp | -48 | 57 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 668 | 277 | 391 | 41.47% | 43.33% | 41.88% | 8.53 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
