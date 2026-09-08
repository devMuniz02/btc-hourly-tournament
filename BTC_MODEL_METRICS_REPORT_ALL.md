# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T05:30:29.943341+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1288 | 1000 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1163 | 798 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 883 | 560 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 885 | 614 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 203 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 203 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 66 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 00:00:00+00:00 | 203 | 66 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 560 | 274 | 286 | 48.93% | 47.50% | 48.12% | 1.07 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 560 | 267 | 293 | 47.68% | 51.67% | 49.58% | 2.32 pp | -26 | 52 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 788 | 380 | 408 | 48.22% | 46.67% | 48.12% | 1.78 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| BTC Market Hours | transformer | Transformer | 560 | 263 | 297 | 46.96% | 46.25% | 47.29% | 3.04 pp | -34 | 52 | -0.65 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | nn | NN | 614 | 287 | 327 | 46.74% | 47.50% | 48.33% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 614 | 287 | 327 | 46.74% | 49.17% | 47.50% | 3.26 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 614 | 286 | 328 | 46.58% | 49.17% | 47.29% | 3.42 pp | -42 | 52 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 966 | 460 | 506 | 47.62% | 49.58% | 46.67% | 2.38 pp | -46 | 50 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| BTC Daily | transformer | Transformer | 788 | 366 | 422 | 46.45% | 40.42% | 46.46% | 3.55 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 788 | 365 | 423 | 46.32% | 45.00% | 45.00% | 3.68 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 966 | 450 | 516 | 46.58% | 44.58% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | lstm | LSTM | 560 | 243 | 317 | 43.39% | 42.08% | 43.96% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | rf | RandomForest | 560 | 243 | 317 | 43.39% | 45.83% | 43.54% | 6.61 pp | -74 | 52 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 560 | 238 | 322 | 42.50% | 45.83% | 42.92% | 7.50 pp | -84 | 52 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 614 | 257 | 357 | 41.86% | 44.17% | 41.04% | 8.14 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 614 | 251 | 363 | 40.88% | 43.33% | 40.62% | 9.12 pp | -112 | 52 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 614 | 250 | 364 | 40.72% | 40.83% | 40.42% | 9.28 pp | -114 | 52 | -2.19 |
| BTC Hourly | rf | RandomForest | 966 | 428 | 538 | 44.31% | 42.08% | 42.71% | 5.69 pp | -110 | 50 | -2.20 |
| BTC Hourly | nn | NN | 966 | 427 | 539 | 44.20% | 41.67% | 42.50% | 5.80 pp | -112 | 50 | -2.24 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| BTC Daily | lstm | LSTM | 788 | 332 | 456 | 42.13% | 34.17% | 40.00% | 7.87 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 788 | 329 | 459 | 41.75% | 38.33% | 41.67% | 8.25 pp | -130 | 46 | -2.83 |
| BTC Hourly | lstm | LSTM | 966 | 412 | 554 | 42.65% | 37.92% | 41.67% | 7.35 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 966 | 399 | 567 | 41.30% | 36.67% | 38.75% | 8.70 pp | -168 | 50 | -3.36 |
| BTC Daily | xgb | XGBoost | 798 | 311 | 487 | 38.97% | 35.42% | 36.25% | 11.03 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 966 | 460 | 506 | 47.62% | 49.58% | 46.67% | 2.38 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 966 | 450 | 516 | 46.58% | 44.58% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 966 | 428 | 538 | 44.31% | 42.08% | 42.71% | 5.69 pp | -110 | 50 | -2.20 |
| BTC Hourly | nn | NN | 966 | 427 | 539 | 44.20% | 41.67% | 42.50% | 5.80 pp | -112 | 50 | -2.24 |
| BTC Hourly | lstm | LSTM | 966 | 412 | 554 | 42.65% | 37.92% | 41.67% | 7.35 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 966 | 399 | 567 | 41.30% | 36.67% | 38.75% | 8.70 pp | -168 | 50 | -3.36 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 788 | 380 | 408 | 48.22% | 46.67% | 48.12% | 1.78 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 788 | 366 | 422 | 46.45% | 40.42% | 46.46% | 3.55 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 788 | 365 | 423 | 46.32% | 45.00% | 45.00% | 3.68 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 788 | 332 | 456 | 42.13% | 34.17% | 40.00% | 7.87 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 788 | 329 | 459 | 41.75% | 38.33% | 41.67% | 8.25 pp | -130 | 46 | -2.83 |
| BTC Daily | xgb | XGBoost | 798 | 311 | 487 | 38.97% | 35.42% | 36.25% | 11.03 pp | -176 | 46 | -3.83 |

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
| Consolidated Hourly | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 90 | 113 | 44.33% | 44.33% | 44.33% | 5.67 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 14 | -2.07 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
