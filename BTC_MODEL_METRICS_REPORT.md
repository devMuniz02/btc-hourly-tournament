# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T03:26:07.625746+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1302 | 1014 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1178 | 813 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 911 | 575 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 913 | 629 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 217 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 217 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 217 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T18:00:00+00:00 | 218 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 575 | 280 | 295 | 48.70% | 47.50% | 47.92% | 1.30 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 575 | 274 | 301 | 47.65% | 51.67% | 49.17% | 2.35 pp | -27 | 54 | -0.50 |
| BTC Market Hours | transformer | Transformer | 575 | 271 | 304 | 47.13% | 47.08% | 46.88% | 2.87 pp | -33 | 54 | -0.61 |
| BTC Daily | mlp_sklearn | MLPClassifier | 803 | 387 | 416 | 48.19% | 46.67% | 47.29% | 1.81 pp | -29 | 46 | -0.63 |
| Consolidated Hourly | rf | RandomForest | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | nn | NN | 629 | 294 | 335 | 46.74% | 47.50% | 47.92% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 629 | 294 | 335 | 46.74% | 49.17% | 47.29% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 629 | 292 | 337 | 46.42% | 47.92% | 46.67% | 3.58 pp | -45 | 54 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 980 | 468 | 512 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 803 | 373 | 430 | 46.45% | 44.58% | 44.79% | 3.55 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 803 | 372 | 431 | 46.33% | 39.58% | 45.83% | 3.67 pp | -59 | 46 | -1.28 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| BTC Hourly | transformer | Transformer | 980 | 454 | 526 | 46.33% | 43.33% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 575 | 249 | 326 | 43.30% | 42.92% | 43.54% | 6.70 pp | -77 | 54 | -1.43 |
| BTC Market Hours | rf | RandomForest | 575 | 248 | 327 | 43.13% | 45.00% | 43.54% | 6.87 pp | -79 | 54 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 575 | 248 | 327 | 43.13% | 46.67% | 43.33% | 6.87 pp | -79 | 54 | -1.46 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 629 | 263 | 366 | 41.81% | 43.75% | 40.62% | 8.19 pp | -103 | 54 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 629 | 261 | 368 | 41.49% | 45.00% | 41.04% | 8.51 pp | -107 | 54 | -1.98 |
| Consolidated Hourly | nn | NN | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 629 | 257 | 372 | 40.86% | 42.08% | 40.21% | 9.14 pp | -115 | 54 | -2.13 |
| BTC Hourly | rf | RandomForest | 980 | 433 | 547 | 44.18% | 41.67% | 42.71% | 5.82 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 980 | 432 | 548 | 44.08% | 41.25% | 42.29% | 5.92 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours Daily | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 803 | 338 | 465 | 42.09% | 35.00% | 40.21% | 7.91 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 980 | 417 | 563 | 42.55% | 37.50% | 41.04% | 7.45 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 803 | 334 | 469 | 41.59% | 37.08% | 41.04% | 8.41 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 980 | 405 | 575 | 41.33% | 36.25% | 39.17% | 8.67 pp | -170 | 51 | -3.33 |
| BTC Daily | xgb | XGBoost | 813 | 316 | 497 | 38.87% | 35.00% | 35.62% | 11.13 pp | -181 | 46 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 980 | 468 | 512 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| BTC Hourly | transformer | Transformer | 980 | 454 | 526 | 46.33% | 43.33% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Hourly | rf | RandomForest | 980 | 433 | 547 | 44.18% | 41.67% | 42.71% | 5.82 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 980 | 432 | 548 | 44.08% | 41.25% | 42.29% | 5.92 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 980 | 417 | 563 | 42.55% | 37.50% | 41.04% | 7.45 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 980 | 405 | 575 | 41.33% | 36.25% | 39.17% | 8.67 pp | -170 | 51 | -3.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 803 | 387 | 416 | 48.19% | 46.67% | 47.29% | 1.81 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 803 | 373 | 430 | 46.45% | 44.58% | 44.79% | 3.55 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 803 | 372 | 431 | 46.33% | 39.58% | 45.83% | 3.67 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 803 | 338 | 465 | 42.09% | 35.00% | 40.21% | 7.91 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 803 | 334 | 469 | 41.59% | 37.08% | 41.04% | 8.41 pp | -135 | 46 | -2.93 |
| BTC Daily | xgb | XGBoost | 813 | 316 | 497 | 38.87% | 35.00% | 35.62% | 11.13 pp | -181 | 46 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 575 | 280 | 295 | 48.70% | 47.50% | 47.92% | 1.30 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 575 | 274 | 301 | 47.65% | 51.67% | 49.17% | 2.35 pp | -27 | 54 | -0.50 |
| BTC Market Hours | transformer | Transformer | 575 | 271 | 304 | 47.13% | 47.08% | 46.88% | 2.87 pp | -33 | 54 | -0.61 |
| BTC Market Hours | lstm | LSTM | 575 | 249 | 326 | 43.30% | 42.92% | 43.54% | 6.70 pp | -77 | 54 | -1.43 |
| BTC Market Hours | rf | RandomForest | 575 | 248 | 327 | 43.13% | 45.00% | 43.54% | 6.87 pp | -79 | 54 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 575 | 248 | 327 | 43.13% | 46.67% | 43.33% | 6.87 pp | -79 | 54 | -1.46 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 629 | 294 | 335 | 46.74% | 47.50% | 47.92% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 629 | 294 | 335 | 46.74% | 49.17% | 47.29% | 3.26 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 629 | 292 | 337 | 46.42% | 47.92% | 46.67% | 3.58 pp | -45 | 54 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 629 | 263 | 366 | 41.81% | 43.75% | 40.62% | 8.19 pp | -103 | 54 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 629 | 261 | 368 | 41.49% | 45.00% | 41.04% | 8.51 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 629 | 257 | 372 | 40.86% | 42.08% | 40.21% | 9.14 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 104 | 113 | 47.93% | 47.93% | 47.93% | 2.07 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 99 | 118 | 45.62% | 45.62% | 45.62% | 4.38 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 94 | 123 | 43.32% | 43.32% | 43.32% | 6.68 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
