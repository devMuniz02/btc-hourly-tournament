# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T19:04:13.392396+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1199 | 911 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1075 | 710 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 18:00:00+00:00 | 724 | 472 | 251 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 18:00:00+00:00 | 726 | 526 | 198 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 122 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 122 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 122 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T18:00:00+00:00 | 123 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 700 | 343 | 357 | 49.00% | 47.50% | 49.17% | 1.00 pp | -14 | 42 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 472 | 228 | 244 | 48.31% | 43.75% | 48.31% | 1.69 pp | -16 | 46 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| BTC Market Hours | nn | NN | 472 | 222 | 250 | 47.03% | 47.50% | 47.03% | 2.97 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 700 | 337 | 363 | 48.14% | 47.50% | 49.79% | 1.86 pp | -26 | 42 | -0.62 |
| BTC Market Hours | transformer | Transformer | 472 | 219 | 253 | 46.40% | 40.83% | 46.40% | 3.60 pp | -34 | 46 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 526 | 242 | 284 | 46.01% | 47.92% | 46.67% | 3.99 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 526 | 241 | 285 | 45.82% | 47.08% | 46.46% | 4.18 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | nn | NN | 526 | 241 | 285 | 45.82% | 43.33% | 46.67% | 4.18 pp | -44 | 46 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 877 | 416 | 461 | 47.43% | 48.75% | 47.92% | 2.57 pp | -45 | 47 | -0.96 |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 877 | 414 | 463 | 47.21% | 47.92% | 47.71% | 2.79 pp | -49 | 47 | -1.04 |
| BTC Daily | nn | NN | 700 | 326 | 374 | 46.57% | 43.75% | 48.75% | 3.43 pp | -48 | 42 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 122 | 55 | 67 | 45.08% | 45.08% | 45.08% | 4.92 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 122 | 55 | 67 | 45.08% | 45.08% | 45.08% | 4.92 pp | -12 | 10 | -1.20 |
| BTC Market Hours | rf | RandomForest | 472 | 203 | 269 | 43.01% | 42.50% | 43.01% | 6.99 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 472 | 202 | 270 | 42.80% | 40.42% | 42.80% | 7.20 pp | -68 | 46 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 122 | 53 | 69 | 43.44% | 43.44% | 43.44% | 6.56 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 122 | 53 | 69 | 43.44% | 43.44% | 43.44% | 6.56 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| BTC Hourly | nn | NN | 877 | 395 | 482 | 45.04% | 46.67% | 43.96% | 4.96 pp | -87 | 47 | -1.85 |
| BTC Market Hours | xgb | XGBoost | 472 | 192 | 280 | 40.68% | 40.00% | 40.68% | 9.32 pp | -88 | 46 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 526 | 218 | 308 | 41.44% | 41.67% | 41.46% | 8.56 pp | -90 | 46 | -1.96 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 877 | 391 | 486 | 44.58% | 45.00% | 44.38% | 5.42 pp | -95 | 47 | -2.02 |
| BTC Daily | lstm | LSTM | 700 | 304 | 396 | 43.43% | 38.75% | 42.29% | 6.57 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 526 | 210 | 316 | 39.92% | 37.08% | 40.83% | 10.08 pp | -106 | 46 | -2.30 |
| BTC Daily | rf | RandomForest | 700 | 301 | 399 | 43.00% | 41.67% | 43.54% | 7.00 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 526 | 209 | 317 | 39.73% | 37.92% | 39.17% | 10.27 pp | -108 | 46 | -2.35 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 877 | 374 | 503 | 42.65% | 38.33% | 41.88% | 7.35 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 877 | 372 | 505 | 42.42% | 42.08% | 43.33% | 7.58 pp | -133 | 47 | -2.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 710 | 282 | 428 | 39.72% | 35.83% | 39.58% | 10.28 pp | -146 | 42 | -3.48 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 4 | 18 | 18.18% | 18.18% | 18.18% | 31.82 pp | -14 | 2 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 877 | 416 | 461 | 47.43% | 48.75% | 47.92% | 2.57 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 877 | 414 | 463 | 47.21% | 47.92% | 47.71% | 2.79 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 877 | 395 | 482 | 45.04% | 46.67% | 43.96% | 4.96 pp | -87 | 47 | -1.85 |
| BTC Hourly | rf | RandomForest | 877 | 391 | 486 | 44.58% | 45.00% | 44.38% | 5.42 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 877 | 374 | 503 | 42.65% | 38.33% | 41.88% | 7.35 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 877 | 372 | 505 | 42.42% | 42.08% | 43.33% | 7.58 pp | -133 | 47 | -2.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 700 | 343 | 357 | 49.00% | 47.50% | 49.17% | 1.00 pp | -14 | 42 | -0.33 |
| BTC Daily | transformer | Transformer | 700 | 337 | 363 | 48.14% | 47.50% | 49.79% | 1.86 pp | -26 | 42 | -0.62 |
| BTC Daily | nn | NN | 700 | 326 | 374 | 46.57% | 43.75% | 48.75% | 3.43 pp | -48 | 42 | -1.14 |
| BTC Daily | lstm | LSTM | 700 | 304 | 396 | 43.43% | 38.75% | 42.29% | 6.57 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 700 | 301 | 399 | 43.00% | 41.67% | 43.54% | 7.00 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 710 | 282 | 428 | 39.72% | 35.83% | 39.58% | 10.28 pp | -146 | 42 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 472 | 228 | 244 | 48.31% | 43.75% | 48.31% | 1.69 pp | -16 | 46 | -0.35 |
| BTC Market Hours | nn | NN | 472 | 222 | 250 | 47.03% | 47.50% | 47.03% | 2.97 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 472 | 219 | 253 | 46.40% | 40.83% | 46.40% | 3.60 pp | -34 | 46 | -0.74 |
| BTC Market Hours | rf | RandomForest | 472 | 203 | 269 | 43.01% | 42.50% | 43.01% | 6.99 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 472 | 202 | 270 | 42.80% | 40.42% | 42.80% | 7.20 pp | -68 | 46 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 472 | 192 | 280 | 40.68% | 40.00% | 40.68% | 9.32 pp | -88 | 46 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 526 | 242 | 284 | 46.01% | 47.92% | 46.67% | 3.99 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 526 | 241 | 285 | 45.82% | 47.08% | 46.46% | 4.18 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | nn | NN | 526 | 241 | 285 | 45.82% | 43.33% | 46.67% | 4.18 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 526 | 218 | 308 | 41.44% | 41.67% | 41.46% | 8.56 pp | -90 | 46 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 526 | 210 | 316 | 39.92% | 37.08% | 40.83% | 10.08 pp | -106 | 46 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 526 | 209 | 317 | 39.73% | 37.92% | 39.17% | 10.27 pp | -108 | 46 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 122 | 55 | 67 | 45.08% | 45.08% | 45.08% | 4.92 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | nn | NN | 122 | 53 | 69 | 43.44% | 43.44% | 43.44% | 6.56 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 122 | 61 | 61 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 122 | 58 | 64 | 47.54% | 47.54% | 47.54% | 2.46 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 122 | 55 | 67 | 45.08% | 45.08% | 45.08% | 4.92 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 122 | 53 | 69 | 43.44% | 43.44% | 43.44% | 6.56 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 4 | 18 | 18.18% | 18.18% | 18.18% | 31.82 pp | -14 | 2 | -7.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
