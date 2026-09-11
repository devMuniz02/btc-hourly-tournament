# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T16:08:53.979017+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1218 | 853 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 981 | 615 | 365 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 983 | 669 | 312 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T12:00:00+00:00 | 253 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T12:00:00+00:00 | 253 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T12:00:00+00:00 | 253 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T12:00:00+00:00 | 254 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 615 | 296 | 319 | 48.13% | 46.67% | 47.29% | 1.87 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 615 | 294 | 321 | 47.80% | 50.42% | 49.58% | 2.20 pp | -27 | 57 | -0.47 |
| BTC Market Hours Daily | nn | NN | 669 | 316 | 353 | 47.23% | 50.42% | 48.75% | 2.77 pp | -37 | 57 | -0.65 |
| BTC Market Hours | transformer | Transformer | 615 | 287 | 328 | 46.67% | 46.25% | 45.62% | 3.33 pp | -41 | 57 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 669 | 313 | 356 | 46.79% | 49.17% | 47.29% | 3.21 pp | -43 | 57 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 843 | 401 | 442 | 47.57% | 44.17% | 46.04% | 2.43 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 669 | 310 | 359 | 46.34% | 47.92% | 47.92% | 3.66 pp | -49 | 57 | -0.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 8 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1020 | 482 | 538 | 47.25% | 47.08% | 46.04% | 2.75 pp | -56 | 52 | -1.08 |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| BTC Daily | nn | NN | 843 | 394 | 449 | 46.74% | 46.25% | 45.42% | 3.26 pp | -55 | 48 | -1.15 |
| Consolidated Hourly | rf | RandomForest | 253 | 117 | 136 | 46.25% | 46.25% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 117 | 136 | 46.25% | 46.25% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1020 | 478 | 542 | 46.86% | 47.08% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| BTC Daily | transformer | Transformer | 843 | 388 | 455 | 46.03% | 37.50% | 44.17% | 3.97 pp | -67 | 48 | -1.40 |
| Consolidated Hourly | lstm | LSTM | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.42% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.42% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 615 | 264 | 351 | 42.93% | 43.33% | 43.12% | 7.07 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 615 | 263 | 352 | 42.76% | 43.33% | 41.67% | 7.24 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 615 | 262 | 353 | 42.60% | 46.67% | 43.12% | 7.40 pp | -91 | 57 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 16 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 16 | -1.69 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 669 | 278 | 391 | 41.55% | 43.33% | 41.88% | 8.45 pp | -113 | 57 | -1.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 669 | 275 | 394 | 41.11% | 44.17% | 41.25% | 8.89 pp | -119 | 57 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 669 | 274 | 395 | 40.96% | 43.75% | 41.04% | 9.04 pp | -121 | 57 | -2.12 |
| Consolidated Hourly | nn | NN | 253 | 108 | 145 | 42.69% | 43.33% | 42.69% | 7.31 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 108 | 145 | 42.69% | 43.33% | 42.69% | 7.31 pp | -37 | 16 | -2.31 |
| BTC Hourly | nn | NN | 1020 | 448 | 572 | 43.92% | 40.42% | 40.62% | 6.08 pp | -124 | 52 | -2.38 |
| BTC Hourly | rf | RandomForest | 1020 | 447 | 573 | 43.82% | 40.83% | 42.29% | 6.18 pp | -126 | 52 | -2.42 |
| Consolidated Hourly | xgb | XGBoost | 253 | 107 | 146 | 42.29% | 42.08% | 42.29% | 7.71 pp | -39 | 16 | -2.44 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 107 | 146 | 42.29% | 42.08% | 42.29% | 7.71 pp | -39 | 16 | -2.44 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 843 | 356 | 487 | 42.23% | 36.25% | 39.79% | 7.77 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 843 | 350 | 493 | 41.52% | 37.08% | 40.83% | 8.48 pp | -143 | 48 | -2.98 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| BTC Hourly | lstm | LSTM | 1020 | 428 | 592 | 41.96% | 34.17% | 39.17% | 8.04 pp | -164 | 52 | -3.15 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| BTC Hourly | xgb | XGBoost | 1020 | 418 | 602 | 40.98% | 34.58% | 37.50% | 9.02 pp | -184 | 52 | -3.54 |
| BTC Daily | xgb | XGBoost | 853 | 337 | 516 | 39.51% | 37.92% | 36.46% | 10.49 pp | -179 | 48 | -3.73 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 843 | 401 | 442 | 47.57% | 44.17% | 46.04% | 2.43 pp | -41 | 48 | -0.85 |
| BTC Daily | nn | NN | 843 | 394 | 449 | 46.74% | 46.25% | 45.42% | 3.26 pp | -55 | 48 | -1.15 |
| BTC Daily | transformer | Transformer | 843 | 388 | 455 | 46.03% | 37.50% | 44.17% | 3.97 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 843 | 356 | 487 | 42.23% | 36.25% | 39.79% | 7.77 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 843 | 350 | 493 | 41.52% | 37.08% | 40.83% | 8.48 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 853 | 337 | 516 | 39.51% | 37.92% | 36.46% | 10.49 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 615 | 296 | 319 | 48.13% | 46.67% | 47.29% | 1.87 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 615 | 294 | 321 | 47.80% | 50.42% | 49.58% | 2.20 pp | -27 | 57 | -0.47 |
| BTC Market Hours | transformer | Transformer | 615 | 287 | 328 | 46.67% | 46.25% | 45.62% | 3.33 pp | -41 | 57 | -0.72 |
| BTC Market Hours | lstm | LSTM | 615 | 264 | 351 | 42.93% | 43.33% | 43.12% | 7.07 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 615 | 263 | 352 | 42.76% | 43.33% | 41.67% | 7.24 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 615 | 262 | 353 | 42.60% | 46.67% | 43.12% | 7.40 pp | -91 | 57 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 669 | 316 | 353 | 47.23% | 50.42% | 48.75% | 2.77 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 669 | 313 | 356 | 46.79% | 49.17% | 47.29% | 3.21 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 669 | 310 | 359 | 46.34% | 47.92% | 47.92% | 3.66 pp | -49 | 57 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 669 | 278 | 391 | 41.55% | 43.33% | 41.88% | 8.45 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 669 | 275 | 394 | 41.11% | 44.17% | 41.25% | 8.89 pp | -119 | 57 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 669 | 274 | 395 | 40.96% | 43.75% | 41.04% | 9.04 pp | -121 | 57 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 253 | 117 | 136 | 46.25% | 46.25% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.42% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 16 | -1.69 |
| Consolidated Hourly | nn | NN | 253 | 108 | 145 | 42.69% | 43.33% | 42.69% | 7.31 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 253 | 107 | 146 | 42.29% | 42.08% | 42.29% | 7.71 pp | -39 | 16 | -2.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 117 | 136 | 46.25% | 46.25% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.42% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 16 | -1.69 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 108 | 145 | 42.69% | 43.33% | 42.69% | 7.31 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 107 | 146 | 42.29% | 42.08% | 42.29% | 7.71 pp | -39 | 16 | -2.44 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
