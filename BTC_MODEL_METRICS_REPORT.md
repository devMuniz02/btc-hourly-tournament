# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T23:33:39.788238+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1347 | 1059 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1223 | 858 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 22:00:00+00:00 | 993 | 620 | 372 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 22:00:00+00:00 | 995 | 674 | 319 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 620 | 298 | 322 | 48.06% | 46.67% | 47.08% | 1.94 pp | -24 | 57 | -0.42 |
| BTC Market Hours | nn | NN | 620 | 296 | 324 | 47.74% | 50.00% | 49.58% | 2.26 pp | -28 | 57 | -0.49 |
| BTC Market Hours Daily | nn | NN | 674 | 318 | 356 | 47.18% | 50.42% | 48.33% | 2.82 pp | -38 | 57 | -0.67 |
| BTC Market Hours | transformer | Transformer | 620 | 289 | 331 | 46.61% | 46.25% | 45.21% | 3.39 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 674 | 315 | 359 | 46.74% | 48.75% | 47.29% | 3.26 pp | -44 | 57 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 674 | 312 | 362 | 46.29% | 47.50% | 47.29% | 3.71 pp | -50 | 57 | -0.88 |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| BTC Daily | mlp_sklearn | MLPClassifier | 848 | 401 | 447 | 47.29% | 42.92% | 45.42% | 2.71 pp | -46 | 48 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1025 | 484 | 541 | 47.22% | 46.67% | 45.83% | 2.78 pp | -57 | 53 | -1.08 |
| BTC Hourly | transformer | Transformer | 1025 | 482 | 543 | 47.02% | 47.50% | 45.21% | 2.98 pp | -61 | 53 | -1.15 |
| BTC Daily | nn | NN | 848 | 396 | 452 | 46.70% | 46.25% | 45.21% | 3.30 pp | -56 | 48 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 848 | 391 | 457 | 46.11% | 38.33% | 44.17% | 3.89 pp | -66 | 48 | -1.38 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | lstm | LSTM | 620 | 266 | 354 | 42.90% | 42.92% | 43.54% | 7.10 pp | -88 | 57 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| BTC Market Hours | rf | RandomForest | 620 | 265 | 355 | 42.74% | 43.75% | 42.08% | 7.26 pp | -90 | 57 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 620 | 263 | 357 | 42.42% | 46.67% | 42.50% | 7.58 pp | -94 | 57 | -1.65 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 674 | 280 | 394 | 41.54% | 43.75% | 41.46% | 8.46 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 674 | 276 | 398 | 40.95% | 43.75% | 40.83% | 9.05 pp | -122 | 57 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 674 | 276 | 398 | 40.95% | 44.17% | 41.25% | 9.05 pp | -122 | 57 | -2.14 |
| BTC Hourly | nn | NN | 1025 | 451 | 574 | 44.00% | 40.83% | 40.62% | 6.00 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1025 | 448 | 577 | 43.71% | 40.83% | 42.08% | 6.29 pp | -129 | 53 | -2.43 |
| BTC Daily | lstm | LSTM | 848 | 358 | 490 | 42.22% | 36.25% | 40.00% | 7.78 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 848 | 353 | 495 | 41.63% | 37.92% | 40.83% | 8.37 pp | -142 | 48 | -2.96 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| BTC Hourly | lstm | LSTM | 1025 | 429 | 596 | 41.85% | 34.17% | 38.96% | 8.15 pp | -167 | 53 | -3.15 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1025 | 421 | 604 | 41.07% | 35.42% | 37.71% | 8.93 pp | -183 | 53 | -3.45 |
| BTC Daily | xgb | XGBoost | 858 | 338 | 520 | 39.39% | 37.92% | 35.83% | 10.61 pp | -182 | 48 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1025 | 484 | 541 | 47.22% | 46.67% | 45.83% | 2.78 pp | -57 | 53 | -1.08 |
| BTC Hourly | transformer | Transformer | 1025 | 482 | 543 | 47.02% | 47.50% | 45.21% | 2.98 pp | -61 | 53 | -1.15 |
| BTC Hourly | nn | NN | 1025 | 451 | 574 | 44.00% | 40.83% | 40.62% | 6.00 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1025 | 448 | 577 | 43.71% | 40.83% | 42.08% | 6.29 pp | -129 | 53 | -2.43 |
| BTC Hourly | lstm | LSTM | 1025 | 429 | 596 | 41.85% | 34.17% | 38.96% | 8.15 pp | -167 | 53 | -3.15 |
| BTC Hourly | xgb | XGBoost | 1025 | 421 | 604 | 41.07% | 35.42% | 37.71% | 8.93 pp | -183 | 53 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 848 | 401 | 447 | 47.29% | 42.92% | 45.42% | 2.71 pp | -46 | 48 | -0.96 |
| BTC Daily | nn | NN | 848 | 396 | 452 | 46.70% | 46.25% | 45.21% | 3.30 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 848 | 391 | 457 | 46.11% | 38.33% | 44.17% | 3.89 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 848 | 358 | 490 | 42.22% | 36.25% | 40.00% | 7.78 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 848 | 353 | 495 | 41.63% | 37.92% | 40.83% | 8.37 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 858 | 338 | 520 | 39.39% | 37.92% | 35.83% | 10.61 pp | -182 | 48 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 620 | 298 | 322 | 48.06% | 46.67% | 47.08% | 1.94 pp | -24 | 57 | -0.42 |
| BTC Market Hours | nn | NN | 620 | 296 | 324 | 47.74% | 50.00% | 49.58% | 2.26 pp | -28 | 57 | -0.49 |
| BTC Market Hours | transformer | Transformer | 620 | 289 | 331 | 46.61% | 46.25% | 45.21% | 3.39 pp | -42 | 57 | -0.74 |
| BTC Market Hours | lstm | LSTM | 620 | 266 | 354 | 42.90% | 42.92% | 43.54% | 7.10 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 620 | 265 | 355 | 42.74% | 43.75% | 42.08% | 7.26 pp | -90 | 57 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 620 | 263 | 357 | 42.42% | 46.67% | 42.50% | 7.58 pp | -94 | 57 | -1.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 674 | 318 | 356 | 47.18% | 50.42% | 48.33% | 2.82 pp | -38 | 57 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 674 | 315 | 359 | 46.74% | 48.75% | 47.29% | 3.26 pp | -44 | 57 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 674 | 312 | 362 | 46.29% | 47.50% | 47.29% | 3.71 pp | -50 | 57 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 674 | 280 | 394 | 41.54% | 43.75% | 41.46% | 8.46 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 674 | 276 | 398 | 40.95% | 43.75% | 40.83% | 9.05 pp | -122 | 57 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 674 | 276 | 398 | 40.95% | 44.17% | 41.25% | 9.05 pp | -122 | 57 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
