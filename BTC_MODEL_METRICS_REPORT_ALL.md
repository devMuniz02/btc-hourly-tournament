# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T04:15:48.153902+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1287 | 999 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1163 | 798 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 883 | 560 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 885 | 614 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 560 | 274 | 286 | 48.93% | 47.50% | 48.12% | 1.07 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 560 | 267 | 293 | 47.68% | 51.67% | 49.58% | 2.32 pp | -26 | 52 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 788 | 380 | 408 | 48.22% | 46.67% | 48.12% | 1.78 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 560 | 263 | 297 | 46.96% | 46.25% | 47.29% | 3.04 pp | -34 | 52 | -0.65 |
| BTC Market Hours Daily | nn | NN | 614 | 287 | 327 | 46.74% | 47.50% | 48.33% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 614 | 287 | 327 | 46.74% | 49.17% | 47.50% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 614 | 286 | 328 | 46.58% | 49.17% | 47.29% | 3.42 pp | -42 | 52 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 965 | 460 | 505 | 47.67% | 49.58% | 46.88% | 2.33 pp | -45 | 50 | -0.90 |
| BTC Daily | transformer | Transformer | 788 | 367 | 421 | 46.57% | 40.83% | 46.67% | 3.43 pp | -54 | 46 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 788 | 365 | 423 | 46.32% | 45.00% | 45.00% | 3.68 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 965 | 449 | 516 | 46.53% | 44.58% | 43.96% | 3.47 pp | -67 | 50 | -1.34 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 560 | 243 | 317 | 43.39% | 42.08% | 43.96% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 560 | 243 | 317 | 43.39% | 45.83% | 43.54% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 560 | 238 | 322 | 42.50% | 45.83% | 42.92% | 7.50 pp | -84 | 52 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 614 | 257 | 357 | 41.86% | 44.17% | 41.04% | 8.14 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 614 | 251 | 363 | 40.88% | 43.33% | 40.62% | 9.12 pp | -112 | 52 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 614 | 250 | 364 | 40.72% | 40.83% | 40.42% | 9.28 pp | -114 | 52 | -2.19 |
| BTC Hourly | nn | NN | 965 | 427 | 538 | 44.25% | 41.67% | 42.71% | 5.75 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 965 | 427 | 538 | 44.25% | 41.67% | 42.50% | 5.75 pp | -111 | 50 | -2.22 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 788 | 331 | 457 | 42.01% | 33.75% | 39.79% | 7.99 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 788 | 330 | 458 | 41.88% | 38.75% | 41.88% | 8.12 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 965 | 411 | 554 | 42.59% | 37.50% | 41.46% | 7.41 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 965 | 398 | 567 | 41.24% | 36.67% | 38.54% | 8.76 pp | -169 | 50 | -3.38 |
| BTC Daily | xgb | XGBoost | 798 | 312 | 486 | 39.10% | 35.83% | 36.46% | 10.90 pp | -174 | 46 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 965 | 460 | 505 | 47.67% | 49.58% | 46.88% | 2.33 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 965 | 449 | 516 | 46.53% | 44.58% | 43.96% | 3.47 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 965 | 427 | 538 | 44.25% | 41.67% | 42.71% | 5.75 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 965 | 427 | 538 | 44.25% | 41.67% | 42.50% | 5.75 pp | -111 | 50 | -2.22 |
| BTC Hourly | lstm | LSTM | 965 | 411 | 554 | 42.59% | 37.50% | 41.46% | 7.41 pp | -143 | 50 | -2.86 |
| BTC Hourly | xgb | XGBoost | 965 | 398 | 567 | 41.24% | 36.67% | 38.54% | 8.76 pp | -169 | 50 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 788 | 380 | 408 | 48.22% | 46.67% | 48.12% | 1.78 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 788 | 367 | 421 | 46.57% | 40.83% | 46.67% | 3.43 pp | -54 | 46 | -1.17 |
| BTC Daily | nn | NN | 788 | 365 | 423 | 46.32% | 45.00% | 45.00% | 3.68 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 788 | 331 | 457 | 42.01% | 33.75% | 39.79% | 7.99 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 788 | 330 | 458 | 41.88% | 38.75% | 41.88% | 8.12 pp | -128 | 46 | -2.78 |
| BTC Daily | xgb | XGBoost | 798 | 312 | 486 | 39.10% | 35.83% | 36.46% | 10.90 pp | -174 | 46 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 560 | 274 | 286 | 48.93% | 47.50% | 48.12% | 1.07 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 560 | 267 | 293 | 47.68% | 51.67% | 49.58% | 2.32 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 560 | 263 | 297 | 46.96% | 46.25% | 47.29% | 3.04 pp | -34 | 52 | -0.65 |
| BTC Market Hours | lstm | LSTM | 560 | 243 | 317 | 43.39% | 42.08% | 43.96% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 560 | 243 | 317 | 43.39% | 45.83% | 43.54% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 560 | 238 | 322 | 42.50% | 45.83% | 42.92% | 7.50 pp | -84 | 52 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 614 | 287 | 327 | 46.74% | 47.50% | 48.33% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 614 | 287 | 327 | 46.74% | 49.17% | 47.50% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 614 | 286 | 328 | 46.58% | 49.17% | 47.29% | 3.42 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 614 | 257 | 357 | 41.86% | 44.17% | 41.04% | 8.14 pp | -100 | 52 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 614 | 251 | 363 | 40.88% | 43.33% | 40.62% | 9.12 pp | -112 | 52 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 614 | 250 | 364 | 40.72% | 40.83% | 40.42% | 9.28 pp | -114 | 52 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
