# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T22:00:58.957571+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1346 | 1058 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1222 | 857 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 21:00:00+00:00 | 991 | 619 | 371 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 21:00:00+00:00 | 993 | 673 | 318 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 619 | 297 | 322 | 47.98% | 46.25% | 47.08% | 2.02 pp | -25 | 57 | -0.44 |
| BTC Market Hours | nn | NN | 619 | 296 | 323 | 47.82% | 50.42% | 49.58% | 2.18 pp | -27 | 57 | -0.47 |
| BTC Market Hours Daily | nn | NN | 673 | 318 | 355 | 47.25% | 50.42% | 48.54% | 2.75 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 673 | 315 | 358 | 46.81% | 48.75% | 47.29% | 3.19 pp | -43 | 57 | -0.75 |
| BTC Market Hours | transformer | Transformer | 619 | 288 | 331 | 46.53% | 45.83% | 45.21% | 3.47 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 673 | 312 | 361 | 46.36% | 47.92% | 47.50% | 3.64 pp | -49 | 57 | -0.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 847 | 401 | 446 | 47.34% | 42.92% | 45.62% | 2.66 pp | -45 | 48 | -0.94 |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1024 | 483 | 541 | 47.17% | 46.25% | 45.83% | 2.83 pp | -58 | 53 | -1.09 |
| BTC Daily | nn | NN | 847 | 396 | 451 | 46.75% | 46.25% | 45.21% | 3.25 pp | -55 | 48 | -1.15 |
| BTC Hourly | transformer | Transformer | 1024 | 481 | 543 | 46.97% | 47.08% | 45.21% | 3.03 pp | -62 | 53 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 847 | 391 | 456 | 46.16% | 38.75% | 44.38% | 3.84 pp | -65 | 48 | -1.35 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | lstm | LSTM | 619 | 265 | 354 | 42.81% | 42.92% | 43.33% | 7.19 pp | -89 | 57 | -1.56 |
| BTC Market Hours | rf | RandomForest | 619 | 265 | 354 | 42.81% | 43.75% | 42.08% | 7.19 pp | -89 | 57 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 619 | 263 | 356 | 42.49% | 46.67% | 42.71% | 7.51 pp | -93 | 57 | -1.63 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 673 | 280 | 393 | 41.60% | 43.75% | 41.46% | 8.40 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 673 | 276 | 397 | 41.01% | 43.75% | 40.83% | 8.99 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 673 | 276 | 397 | 41.01% | 44.17% | 41.25% | 8.99 pp | -121 | 57 | -2.12 |
| BTC Hourly | nn | NN | 1024 | 450 | 574 | 43.95% | 40.42% | 40.42% | 6.05 pp | -124 | 53 | -2.34 |
| BTC Hourly | rf | RandomForest | 1024 | 447 | 577 | 43.65% | 40.42% | 41.88% | 6.35 pp | -130 | 53 | -2.45 |
| BTC Daily | lstm | LSTM | 847 | 357 | 490 | 42.15% | 36.25% | 39.79% | 7.85 pp | -133 | 48 | -2.77 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 847 | 353 | 494 | 41.68% | 37.92% | 40.83% | 8.32 pp | -141 | 48 | -2.94 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| BTC Hourly | lstm | LSTM | 1024 | 429 | 595 | 41.89% | 34.17% | 38.96% | 8.11 pp | -166 | 53 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1024 | 420 | 604 | 41.02% | 35.00% | 37.71% | 8.98 pp | -184 | 53 | -3.47 |
| BTC Daily | xgb | XGBoost | 857 | 338 | 519 | 39.44% | 37.92% | 36.04% | 10.56 pp | -181 | 48 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1024 | 483 | 541 | 47.17% | 46.25% | 45.83% | 2.83 pp | -58 | 53 | -1.09 |
| BTC Hourly | transformer | Transformer | 1024 | 481 | 543 | 46.97% | 47.08% | 45.21% | 3.03 pp | -62 | 53 | -1.17 |
| BTC Hourly | nn | NN | 1024 | 450 | 574 | 43.95% | 40.42% | 40.42% | 6.05 pp | -124 | 53 | -2.34 |
| BTC Hourly | rf | RandomForest | 1024 | 447 | 577 | 43.65% | 40.42% | 41.88% | 6.35 pp | -130 | 53 | -2.45 |
| BTC Hourly | lstm | LSTM | 1024 | 429 | 595 | 41.89% | 34.17% | 38.96% | 8.11 pp | -166 | 53 | -3.13 |
| BTC Hourly | xgb | XGBoost | 1024 | 420 | 604 | 41.02% | 35.00% | 37.71% | 8.98 pp | -184 | 53 | -3.47 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 847 | 401 | 446 | 47.34% | 42.92% | 45.62% | 2.66 pp | -45 | 48 | -0.94 |
| BTC Daily | nn | NN | 847 | 396 | 451 | 46.75% | 46.25% | 45.21% | 3.25 pp | -55 | 48 | -1.15 |
| BTC Daily | transformer | Transformer | 847 | 391 | 456 | 46.16% | 38.75% | 44.38% | 3.84 pp | -65 | 48 | -1.35 |
| BTC Daily | lstm | LSTM | 847 | 357 | 490 | 42.15% | 36.25% | 39.79% | 7.85 pp | -133 | 48 | -2.77 |
| BTC Daily | rf | RandomForest | 847 | 353 | 494 | 41.68% | 37.92% | 40.83% | 8.32 pp | -141 | 48 | -2.94 |
| BTC Daily | xgb | XGBoost | 857 | 338 | 519 | 39.44% | 37.92% | 36.04% | 10.56 pp | -181 | 48 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 619 | 297 | 322 | 47.98% | 46.25% | 47.08% | 2.02 pp | -25 | 57 | -0.44 |
| BTC Market Hours | nn | NN | 619 | 296 | 323 | 47.82% | 50.42% | 49.58% | 2.18 pp | -27 | 57 | -0.47 |
| BTC Market Hours | transformer | Transformer | 619 | 288 | 331 | 46.53% | 45.83% | 45.21% | 3.47 pp | -43 | 57 | -0.75 |
| BTC Market Hours | lstm | LSTM | 619 | 265 | 354 | 42.81% | 42.92% | 43.33% | 7.19 pp | -89 | 57 | -1.56 |
| BTC Market Hours | rf | RandomForest | 619 | 265 | 354 | 42.81% | 43.75% | 42.08% | 7.19 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 619 | 263 | 356 | 42.49% | 46.67% | 42.71% | 7.51 pp | -93 | 57 | -1.63 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 673 | 318 | 355 | 47.25% | 50.42% | 48.54% | 2.75 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 673 | 315 | 358 | 46.81% | 48.75% | 47.29% | 3.19 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 673 | 312 | 361 | 46.36% | 47.92% | 47.50% | 3.64 pp | -49 | 57 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 673 | 280 | 393 | 41.60% | 43.75% | 41.46% | 8.40 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 673 | 276 | 397 | 41.01% | 43.75% | 40.83% | 8.99 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 673 | 276 | 397 | 41.01% | 44.17% | 41.25% | 8.99 pp | -121 | 57 | -2.12 |

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
