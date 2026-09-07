# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T00:58:37.025597+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1268 | 980 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1144 | 779 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 850 | 541 | 308 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 852 | 595 | 255 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 185 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 541 | 262 | 279 | 48.43% | 46.25% | 47.92% | 1.57 pp | -17 | 51 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 541 | 258 | 283 | 47.69% | 48.75% | 48.12% | 2.31 pp | -25 | 51 | -0.49 |
| BTC Daily | mlp_sklearn | MLPClassifier | 769 | 372 | 397 | 48.37% | 46.67% | 48.33% | 1.63 pp | -25 | 45 | -0.56 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 595 | 282 | 313 | 47.39% | 50.83% | 48.75% | 2.61 pp | -31 | 51 | -0.61 |
| BTC Market Hours | nn | NN | 541 | 255 | 286 | 47.13% | 49.58% | 48.96% | 2.87 pp | -31 | 51 | -0.61 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 946 | 453 | 493 | 47.89% | 50.42% | 47.50% | 2.11 pp | -40 | 49 | -0.82 |
| BTC Market Hours Daily | nn | NN | 595 | 276 | 319 | 46.39% | 46.25% | 47.92% | 3.61 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 595 | 275 | 320 | 46.22% | 50.00% | 46.88% | 3.78 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| BTC Daily | transformer | Transformer | 769 | 360 | 409 | 46.81% | 41.67% | 47.08% | 3.19 pp | -49 | 45 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 769 | 357 | 412 | 46.42% | 45.00% | 46.04% | 3.58 pp | -55 | 45 | -1.22 |
| BTC Hourly | transformer | Transformer | 946 | 442 | 504 | 46.72% | 46.25% | 45.00% | 3.28 pp | -62 | 49 | -1.27 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| BTC Market Hours | rf | RandomForest | 541 | 233 | 308 | 43.07% | 44.58% | 43.33% | 6.93 pp | -75 | 51 | -1.47 |
| BTC Market Hours | lstm | LSTM | 541 | 232 | 309 | 42.88% | 41.25% | 43.75% | 7.12 pp | -77 | 51 | -1.51 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 541 | 224 | 317 | 41.40% | 42.92% | 41.88% | 8.60 pp | -93 | 51 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 595 | 248 | 347 | 41.68% | 45.00% | 41.25% | 8.32 pp | -99 | 51 | -1.94 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Hourly | nn | NN | 946 | 420 | 526 | 44.40% | 42.50% | 42.92% | 5.60 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Hourly | rf | RandomForest | 946 | 419 | 527 | 44.29% | 44.17% | 43.75% | 5.71 pp | -108 | 49 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 595 | 240 | 355 | 40.34% | 38.75% | 40.00% | 9.66 pp | -115 | 51 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 595 | 237 | 358 | 39.83% | 41.25% | 38.96% | 10.17 pp | -121 | 51 | -2.37 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 769 | 325 | 444 | 42.26% | 35.83% | 40.21% | 7.74 pp | -119 | 45 | -2.64 |
| BTC Daily | rf | RandomForest | 769 | 323 | 446 | 42.00% | 38.75% | 42.29% | 8.00 pp | -123 | 45 | -2.73 |
| BTC Hourly | lstm | LSTM | 946 | 405 | 541 | 42.81% | 37.08% | 42.29% | 7.19 pp | -136 | 49 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | xgb | XGBoost | 946 | 396 | 550 | 41.86% | 40.42% | 40.62% | 8.14 pp | -154 | 49 | -3.14 |
| BTC Daily | xgb | XGBoost | 779 | 305 | 474 | 39.15% | 35.00% | 36.46% | 10.85 pp | -169 | 45 | -3.76 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 946 | 453 | 493 | 47.89% | 50.42% | 47.50% | 2.11 pp | -40 | 49 | -0.82 |
| BTC Hourly | transformer | Transformer | 946 | 442 | 504 | 46.72% | 46.25% | 45.00% | 3.28 pp | -62 | 49 | -1.27 |
| BTC Hourly | nn | NN | 946 | 420 | 526 | 44.40% | 42.50% | 42.92% | 5.60 pp | -106 | 49 | -2.16 |
| BTC Hourly | rf | RandomForest | 946 | 419 | 527 | 44.29% | 44.17% | 43.75% | 5.71 pp | -108 | 49 | -2.20 |
| BTC Hourly | lstm | LSTM | 946 | 405 | 541 | 42.81% | 37.08% | 42.29% | 7.19 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 946 | 396 | 550 | 41.86% | 40.42% | 40.62% | 8.14 pp | -154 | 49 | -3.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 769 | 372 | 397 | 48.37% | 46.67% | 48.33% | 1.63 pp | -25 | 45 | -0.56 |
| BTC Daily | transformer | Transformer | 769 | 360 | 409 | 46.81% | 41.67% | 47.08% | 3.19 pp | -49 | 45 | -1.09 |
| BTC Daily | nn | NN | 769 | 357 | 412 | 46.42% | 45.00% | 46.04% | 3.58 pp | -55 | 45 | -1.22 |
| BTC Daily | lstm | LSTM | 769 | 325 | 444 | 42.26% | 35.83% | 40.21% | 7.74 pp | -119 | 45 | -2.64 |
| BTC Daily | rf | RandomForest | 769 | 323 | 446 | 42.00% | 38.75% | 42.29% | 8.00 pp | -123 | 45 | -2.73 |
| BTC Daily | xgb | XGBoost | 779 | 305 | 474 | 39.15% | 35.00% | 36.46% | 10.85 pp | -169 | 45 | -3.76 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 541 | 262 | 279 | 48.43% | 46.25% | 47.92% | 1.57 pp | -17 | 51 | -0.33 |
| BTC Market Hours | transformer | Transformer | 541 | 258 | 283 | 47.69% | 48.75% | 48.12% | 2.31 pp | -25 | 51 | -0.49 |
| BTC Market Hours | nn | NN | 541 | 255 | 286 | 47.13% | 49.58% | 48.96% | 2.87 pp | -31 | 51 | -0.61 |
| BTC Market Hours | rf | RandomForest | 541 | 233 | 308 | 43.07% | 44.58% | 43.33% | 6.93 pp | -75 | 51 | -1.47 |
| BTC Market Hours | lstm | LSTM | 541 | 232 | 309 | 42.88% | 41.25% | 43.75% | 7.12 pp | -77 | 51 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 541 | 224 | 317 | 41.40% | 42.92% | 41.88% | 8.60 pp | -93 | 51 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 595 | 282 | 313 | 47.39% | 50.83% | 48.75% | 2.61 pp | -31 | 51 | -0.61 |
| BTC Market Hours Daily | nn | NN | 595 | 276 | 319 | 46.39% | 46.25% | 47.92% | 3.61 pp | -43 | 51 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 595 | 275 | 320 | 46.22% | 50.00% | 46.88% | 3.78 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 595 | 248 | 347 | 41.68% | 45.00% | 41.25% | 8.32 pp | -99 | 51 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 595 | 240 | 355 | 40.34% | 38.75% | 40.00% | 9.66 pp | -115 | 51 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 595 | 237 | 358 | 39.83% | 41.25% | 38.96% | 10.17 pp | -121 | 51 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 90 | 95 | 48.65% | 48.65% | 48.65% | 1.35 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 85 | 100 | 45.95% | 45.95% | 45.95% | 4.05 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
