# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T01:57:07.255451+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1333 | 1045 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1208 | 843 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 967 | 605 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 969 | 659 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 605 | 290 | 315 | 47.93% | 45.83% | 47.29% | 2.07 pp | -25 | 56 | -0.45 |
| BTC Market Hours | nn | NN | 605 | 288 | 317 | 47.60% | 50.42% | 49.17% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 605 | 283 | 322 | 46.78% | 45.83% | 46.25% | 3.22 pp | -39 | 56 | -0.70 |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 659 | 308 | 351 | 46.74% | 48.33% | 47.08% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | nn | NN | 659 | 308 | 351 | 46.74% | 48.33% | 47.71% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 833 | 397 | 436 | 47.66% | 45.00% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 659 | 306 | 353 | 46.43% | 47.92% | 47.50% | 3.57 pp | -47 | 56 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1011 | 478 | 533 | 47.28% | 47.92% | 45.62% | 2.72 pp | -55 | 52 | -1.06 |
| BTC Daily | nn | NN | 833 | 387 | 446 | 46.46% | 45.00% | 45.00% | 3.54 pp | -59 | 48 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 1011 | 471 | 540 | 46.59% | 45.83% | 44.38% | 3.41 pp | -69 | 52 | -1.33 |
| BTC Daily | transformer | Transformer | 833 | 383 | 450 | 45.98% | 37.50% | 44.38% | 4.02 pp | -67 | 48 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 605 | 259 | 346 | 42.81% | 42.50% | 42.92% | 7.19 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 605 | 258 | 347 | 42.64% | 42.92% | 42.29% | 7.36 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 605 | 257 | 348 | 42.48% | 45.00% | 43.33% | 7.52 pp | -91 | 56 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 659 | 272 | 387 | 41.27% | 42.50% | 41.25% | 8.73 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 659 | 269 | 390 | 40.82% | 42.92% | 40.42% | 9.18 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 659 | 269 | 390 | 40.82% | 42.50% | 40.62% | 9.18 pp | -121 | 56 | -2.16 |
| BTC Hourly | nn | NN | 1011 | 446 | 565 | 44.11% | 42.50% | 41.04% | 5.89 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1011 | 445 | 566 | 44.02% | 41.25% | 43.12% | 5.98 pp | -121 | 52 | -2.33 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 833 | 353 | 480 | 42.38% | 36.25% | 40.42% | 7.62 pp | -127 | 48 | -2.65 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| BTC Hourly | lstm | LSTM | 1011 | 427 | 584 | 42.24% | 36.25% | 39.58% | 7.76 pp | -157 | 52 | -3.02 |
| BTC Daily | rf | RandomForest | 833 | 344 | 489 | 41.30% | 36.67% | 40.42% | 8.70 pp | -145 | 48 | -3.02 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1011 | 415 | 596 | 41.05% | 35.00% | 38.12% | 8.95 pp | -181 | 52 | -3.48 |
| BTC Daily | xgb | XGBoost | 843 | 331 | 512 | 39.26% | 37.08% | 36.46% | 10.74 pp | -181 | 48 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1011 | 478 | 533 | 47.28% | 47.92% | 45.62% | 2.72 pp | -55 | 52 | -1.06 |
| BTC Hourly | transformer | Transformer | 1011 | 471 | 540 | 46.59% | 45.83% | 44.38% | 3.41 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1011 | 446 | 565 | 44.11% | 42.50% | 41.04% | 5.89 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1011 | 445 | 566 | 44.02% | 41.25% | 43.12% | 5.98 pp | -121 | 52 | -2.33 |
| BTC Hourly | lstm | LSTM | 1011 | 427 | 584 | 42.24% | 36.25% | 39.58% | 7.76 pp | -157 | 52 | -3.02 |
| BTC Hourly | xgb | XGBoost | 1011 | 415 | 596 | 41.05% | 35.00% | 38.12% | 8.95 pp | -181 | 52 | -3.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 833 | 397 | 436 | 47.66% | 45.00% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Daily | nn | NN | 833 | 387 | 446 | 46.46% | 45.00% | 45.00% | 3.54 pp | -59 | 48 | -1.23 |
| BTC Daily | transformer | Transformer | 833 | 383 | 450 | 45.98% | 37.50% | 44.38% | 4.02 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 833 | 353 | 480 | 42.38% | 36.25% | 40.42% | 7.62 pp | -127 | 48 | -2.65 |
| BTC Daily | rf | RandomForest | 833 | 344 | 489 | 41.30% | 36.67% | 40.42% | 8.70 pp | -145 | 48 | -3.02 |
| BTC Daily | xgb | XGBoost | 843 | 331 | 512 | 39.26% | 37.08% | 36.46% | 10.74 pp | -181 | 48 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 605 | 290 | 315 | 47.93% | 45.83% | 47.29% | 2.07 pp | -25 | 56 | -0.45 |
| BTC Market Hours | nn | NN | 605 | 288 | 317 | 47.60% | 50.42% | 49.17% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 605 | 283 | 322 | 46.78% | 45.83% | 46.25% | 3.22 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 605 | 259 | 346 | 42.81% | 42.50% | 42.92% | 7.19 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 605 | 258 | 347 | 42.64% | 42.92% | 42.29% | 7.36 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 605 | 257 | 348 | 42.48% | 45.00% | 43.33% | 7.52 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 659 | 308 | 351 | 46.74% | 48.33% | 47.08% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | nn | NN | 659 | 308 | 351 | 46.74% | 48.33% | 47.71% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 659 | 306 | 353 | 46.43% | 47.92% | 47.50% | 3.57 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 659 | 272 | 387 | 41.27% | 42.50% | 41.25% | 8.73 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 659 | 269 | 390 | 40.82% | 42.92% | 40.42% | 9.18 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 659 | 269 | 390 | 40.82% | 42.50% | 40.62% | 9.18 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
