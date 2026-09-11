# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T14:19:41.687600+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1341 | 1053 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1217 | 852 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 978 | 614 | 363 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 980 | 668 | 310 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 614 | 296 | 318 | 48.21% | 47.08% | 47.50% | 1.79 pp | -22 | 57 | -0.39 |
| BTC Market Hours | nn | NN | 614 | 294 | 320 | 47.88% | 50.42% | 49.58% | 2.12 pp | -26 | 57 | -0.46 |
| BTC Market Hours Daily | nn | NN | 668 | 315 | 353 | 47.16% | 50.42% | 48.54% | 2.84 pp | -38 | 57 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 668 | 313 | 355 | 46.86% | 49.17% | 47.50% | 3.14 pp | -42 | 57 | -0.74 |
| BTC Market Hours | transformer | Transformer | 614 | 286 | 328 | 46.58% | 46.25% | 45.42% | 3.42 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 668 | 310 | 358 | 46.41% | 47.92% | 48.12% | 3.59 pp | -48 | 57 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 842 | 400 | 442 | 47.51% | 43.75% | 45.83% | 2.49 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1019 | 482 | 537 | 47.30% | 47.50% | 46.04% | 2.70 pp | -55 | 52 | -1.06 |
| BTC Daily | nn | NN | 842 | 394 | 448 | 46.79% | 46.25% | 45.62% | 3.21 pp | -54 | 48 | -1.12 |
| BTC Hourly | transformer | Transformer | 1019 | 478 | 541 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 52 | -1.21 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 842 | 388 | 454 | 46.08% | 37.92% | 44.38% | 3.92 pp | -66 | 48 | -1.38 |
| BTC Market Hours | lstm | LSTM | 614 | 264 | 350 | 43.00% | 43.33% | 43.33% | 7.00 pp | -86 | 57 | -1.51 |
| BTC Market Hours | rf | RandomForest | 614 | 263 | 351 | 42.83% | 43.33% | 41.88% | 7.17 pp | -88 | 57 | -1.54 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 614 | 261 | 353 | 42.51% | 46.67% | 42.92% | 7.49 pp | -92 | 57 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 668 | 277 | 391 | 41.47% | 43.33% | 41.88% | 8.53 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 668 | 274 | 394 | 41.02% | 44.17% | 41.25% | 8.98 pp | -120 | 57 | -2.11 |
| BTC Hourly | nn | NN | 1019 | 448 | 571 | 43.96% | 40.83% | 40.83% | 6.04 pp | -123 | 52 | -2.37 |
| BTC Hourly | rf | RandomForest | 1019 | 446 | 573 | 43.77% | 40.42% | 42.08% | 6.23 pp | -127 | 52 | -2.44 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| BTC Daily | lstm | LSTM | 842 | 355 | 487 | 42.16% | 35.83% | 39.58% | 7.84 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| BTC Daily | rf | RandomForest | 842 | 350 | 492 | 41.57% | 37.08% | 40.83% | 8.43 pp | -142 | 48 | -2.96 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1019 | 428 | 591 | 42.00% | 34.58% | 39.17% | 8.00 pp | -163 | 52 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| BTC Hourly | xgb | XGBoost | 1019 | 417 | 602 | 40.92% | 34.58% | 37.50% | 9.08 pp | -185 | 52 | -3.56 |
| BTC Daily | xgb | XGBoost | 852 | 337 | 515 | 39.55% | 37.92% | 36.67% | 10.45 pp | -178 | 48 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1019 | 482 | 537 | 47.30% | 47.50% | 46.04% | 2.70 pp | -55 | 52 | -1.06 |
| BTC Hourly | transformer | Transformer | 1019 | 478 | 541 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 52 | -1.21 |
| BTC Hourly | nn | NN | 1019 | 448 | 571 | 43.96% | 40.83% | 40.83% | 6.04 pp | -123 | 52 | -2.37 |
| BTC Hourly | rf | RandomForest | 1019 | 446 | 573 | 43.77% | 40.42% | 42.08% | 6.23 pp | -127 | 52 | -2.44 |
| BTC Hourly | lstm | LSTM | 1019 | 428 | 591 | 42.00% | 34.58% | 39.17% | 8.00 pp | -163 | 52 | -3.13 |
| BTC Hourly | xgb | XGBoost | 1019 | 417 | 602 | 40.92% | 34.58% | 37.50% | 9.08 pp | -185 | 52 | -3.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 842 | 400 | 442 | 47.51% | 43.75% | 45.83% | 2.49 pp | -42 | 48 | -0.88 |
| BTC Daily | nn | NN | 842 | 394 | 448 | 46.79% | 46.25% | 45.62% | 3.21 pp | -54 | 48 | -1.12 |
| BTC Daily | transformer | Transformer | 842 | 388 | 454 | 46.08% | 37.92% | 44.38% | 3.92 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 842 | 355 | 487 | 42.16% | 35.83% | 39.58% | 7.84 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 842 | 350 | 492 | 41.57% | 37.08% | 40.83% | 8.43 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 852 | 337 | 515 | 39.55% | 37.92% | 36.67% | 10.45 pp | -178 | 48 | -3.71 |

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
