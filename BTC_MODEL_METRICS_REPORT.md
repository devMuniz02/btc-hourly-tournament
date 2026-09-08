# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T04:52:27.333437+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 204 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 560 | 274 | 286 | 48.93% | 47.50% | 48.12% | 1.07 pp | -12 | 52 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 560 | 267 | 293 | 47.68% | 51.67% | 49.58% | 2.32 pp | -26 | 52 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 788 | 380 | 408 | 48.22% | 46.67% | 48.12% | 1.78 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| BTC Market Hours | transformer | Transformer | 560 | 263 | 297 | 46.96% | 46.25% | 47.29% | 3.04 pp | -34 | 52 | -0.65 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | nn | NN | 614 | 287 | 327 | 46.74% | 47.50% | 48.33% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 614 | 287 | 327 | 46.74% | 49.17% | 47.50% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 614 | 286 | 328 | 46.58% | 49.17% | 47.29% | 3.42 pp | -42 | 52 | -0.81 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 965 | 460 | 505 | 47.67% | 49.58% | 46.88% | 2.33 pp | -45 | 50 | -0.90 |
| BTC Daily | transformer | Transformer | 788 | 367 | 421 | 46.57% | 40.83% | 46.67% | 3.43 pp | -54 | 46 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 788 | 365 | 423 | 46.32% | 45.00% | 45.00% | 3.68 pp | -58 | 46 | -1.26 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 965 | 449 | 516 | 46.53% | 44.58% | 43.96% | 3.47 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 560 | 243 | 317 | 43.39% | 42.08% | 43.96% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 560 | 243 | 317 | 43.39% | 45.83% | 43.54% | 6.61 pp | -74 | 52 | -1.42 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 560 | 238 | 322 | 42.50% | 45.83% | 42.92% | 7.50 pp | -84 | 52 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 614 | 257 | 357 | 41.86% | 44.17% | 41.04% | 8.14 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 614 | 251 | 363 | 40.88% | 43.33% | 40.62% | 9.12 pp | -112 | 52 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 614 | 250 | 364 | 40.72% | 40.83% | 40.42% | 9.28 pp | -114 | 52 | -2.19 |
| BTC Hourly | nn | NN | 965 | 427 | 538 | 44.25% | 41.67% | 42.71% | 5.75 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 965 | 427 | 538 | 44.25% | 41.67% | 42.50% | 5.75 pp | -111 | 50 | -2.22 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
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
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
