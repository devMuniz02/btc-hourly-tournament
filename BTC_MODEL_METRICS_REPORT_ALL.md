# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T06:44:07.839640+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1336 | 1048 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1212 | 847 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 971 | 609 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 972 | 662 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 609 | 294 | 315 | 48.28% | 47.08% | 47.92% | 1.72 pp | -21 | 56 | -0.38 |
| BTC Market Hours | nn | NN | 609 | 291 | 318 | 47.78% | 50.42% | 49.38% | 2.22 pp | -27 | 56 | -0.48 |
| BTC Market Hours | transformer | Transformer | 609 | 285 | 324 | 46.80% | 46.25% | 45.83% | 3.20 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | nn | NN | 662 | 311 | 351 | 46.98% | 49.17% | 47.92% | 3.02 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 662 | 310 | 352 | 46.83% | 48.75% | 47.08% | 3.17 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 662 | 308 | 354 | 46.53% | 48.33% | 47.71% | 3.47 pp | -46 | 56 | -0.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 837 | 398 | 439 | 47.55% | 44.58% | 45.83% | 2.45 pp | -41 | 48 | -0.85 |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1014 | 479 | 535 | 47.24% | 47.92% | 45.62% | 2.76 pp | -56 | 52 | -1.08 |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 837 | 390 | 447 | 46.59% | 45.42% | 45.42% | 3.41 pp | -57 | 48 | -1.19 |
| BTC Hourly | transformer | Transformer | 1014 | 473 | 541 | 46.65% | 46.25% | 44.38% | 3.35 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 837 | 386 | 451 | 46.12% | 37.92% | 44.38% | 3.88 pp | -65 | 48 | -1.35 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 609 | 261 | 348 | 42.86% | 43.33% | 43.12% | 7.14 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 609 | 261 | 348 | 42.86% | 43.75% | 42.29% | 7.14 pp | -87 | 56 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 609 | 259 | 350 | 42.53% | 45.83% | 43.33% | 7.47 pp | -91 | 56 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 662 | 274 | 388 | 41.39% | 42.92% | 41.46% | 8.61 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 662 | 271 | 391 | 40.94% | 43.75% | 40.83% | 9.06 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 662 | 270 | 392 | 40.79% | 42.50% | 40.62% | 9.21 pp | -122 | 56 | -2.18 |
| BTC Hourly | nn | NN | 1014 | 447 | 567 | 44.08% | 42.08% | 40.83% | 5.92 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1014 | 445 | 569 | 43.89% | 40.42% | 42.50% | 6.11 pp | -124 | 52 | -2.38 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 837 | 353 | 484 | 42.17% | 35.83% | 40.00% | 7.83 pp | -131 | 48 | -2.73 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| BTC Daily | rf | RandomForest | 837 | 348 | 489 | 41.58% | 37.08% | 41.04% | 8.42 pp | -141 | 48 | -2.94 |
| BTC Hourly | lstm | LSTM | 1014 | 428 | 586 | 42.21% | 35.83% | 39.38% | 7.79 pp | -158 | 52 | -3.04 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1014 | 415 | 599 | 40.93% | 34.58% | 37.71% | 9.07 pp | -184 | 52 | -3.54 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 847 | 334 | 513 | 39.43% | 37.08% | 36.67% | 10.57 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1014 | 479 | 535 | 47.24% | 47.92% | 45.62% | 2.76 pp | -56 | 52 | -1.08 |
| BTC Hourly | transformer | Transformer | 1014 | 473 | 541 | 46.65% | 46.25% | 44.38% | 3.35 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1014 | 447 | 567 | 44.08% | 42.08% | 40.83% | 5.92 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1014 | 445 | 569 | 43.89% | 40.42% | 42.50% | 6.11 pp | -124 | 52 | -2.38 |
| BTC Hourly | lstm | LSTM | 1014 | 428 | 586 | 42.21% | 35.83% | 39.38% | 7.79 pp | -158 | 52 | -3.04 |
| BTC Hourly | xgb | XGBoost | 1014 | 415 | 599 | 40.93% | 34.58% | 37.71% | 9.07 pp | -184 | 52 | -3.54 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 837 | 398 | 439 | 47.55% | 44.58% | 45.83% | 2.45 pp | -41 | 48 | -0.85 |
| BTC Daily | nn | NN | 837 | 390 | 447 | 46.59% | 45.42% | 45.42% | 3.41 pp | -57 | 48 | -1.19 |
| BTC Daily | transformer | Transformer | 837 | 386 | 451 | 46.12% | 37.92% | 44.38% | 3.88 pp | -65 | 48 | -1.35 |
| BTC Daily | lstm | LSTM | 837 | 353 | 484 | 42.17% | 35.83% | 40.00% | 7.83 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 837 | 348 | 489 | 41.58% | 37.08% | 41.04% | 8.42 pp | -141 | 48 | -2.94 |
| BTC Daily | xgb | XGBoost | 847 | 334 | 513 | 39.43% | 37.08% | 36.67% | 10.57 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 609 | 294 | 315 | 48.28% | 47.08% | 47.92% | 1.72 pp | -21 | 56 | -0.38 |
| BTC Market Hours | nn | NN | 609 | 291 | 318 | 47.78% | 50.42% | 49.38% | 2.22 pp | -27 | 56 | -0.48 |
| BTC Market Hours | transformer | Transformer | 609 | 285 | 324 | 46.80% | 46.25% | 45.83% | 3.20 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 609 | 261 | 348 | 42.86% | 43.33% | 43.12% | 7.14 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 609 | 261 | 348 | 42.86% | 43.75% | 42.29% | 7.14 pp | -87 | 56 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 609 | 259 | 350 | 42.53% | 45.83% | 43.33% | 7.47 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 662 | 311 | 351 | 46.98% | 49.17% | 47.92% | 3.02 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 662 | 310 | 352 | 46.83% | 48.75% | 47.08% | 3.17 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 662 | 308 | 354 | 46.53% | 48.33% | 47.71% | 3.47 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 662 | 274 | 388 | 41.39% | 42.92% | 41.46% | 8.61 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 662 | 271 | 391 | 40.94% | 43.75% | 40.83% | 9.06 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 662 | 270 | 392 | 40.79% | 42.50% | 40.62% | 9.21 pp | -122 | 56 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
