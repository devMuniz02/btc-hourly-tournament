# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T21:52:24.518919+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1169 | 881 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1044 | 679 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 20:00:00+00:00 | 669 | 441 | 227 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 20:00:00+00:00 | 671 | 495 | 174 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 93 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 93 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 6 | 87 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 6 | 87 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 441 | 215 | 226 | 48.75% | 44.58% | 48.75% | 1.25 pp | -11 | 43 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 669 | 326 | 343 | 48.73% | 46.67% | 49.58% | 1.27 pp | -17 | 41 | -0.41 |
| Consolidated Hourly | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| BTC Daily | transformer | Transformer | 669 | 323 | 346 | 48.28% | 45.83% | 49.58% | 1.72 pp | -23 | 41 | -0.56 |
| BTC Market Hours | nn | NN | 441 | 208 | 233 | 47.17% | 48.75% | 47.17% | 2.83 pp | -25 | 43 | -0.58 |
| Consolidated Hourly | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 441 | 202 | 239 | 45.80% | 40.83% | 45.80% | 4.20 pp | -37 | 43 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 495 | 228 | 267 | 46.06% | 46.67% | 46.46% | 3.94 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | nn | NN | 495 | 227 | 268 | 45.86% | 44.17% | 46.46% | 4.14 pp | -41 | 43 | -0.95 |
| Consolidated Hourly | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |
| BTC Hourly | transformer | Transformer | 847 | 400 | 447 | 47.23% | 47.50% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 669 | 313 | 356 | 46.79% | 43.33% | 49.38% | 3.21 pp | -43 | 41 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 495 | 224 | 271 | 45.25% | 45.00% | 45.21% | 4.75 pp | -47 | 43 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 847 | 398 | 449 | 46.99% | 44.58% | 46.88% | 3.01 pp | -51 | 45 | -1.13 |
| BTC Market Hours | lstm | LSTM | 441 | 189 | 252 | 42.86% | 41.67% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 441 | 189 | 252 | 42.86% | 43.33% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Hourly | nn | NN | 847 | 382 | 465 | 45.10% | 43.75% | 44.58% | 4.90 pp | -83 | 45 | -1.84 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 495 | 204 | 291 | 41.21% | 41.25% | 41.46% | 8.79 pp | -87 | 43 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 441 | 177 | 264 | 40.14% | 38.75% | 40.14% | 9.86 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 669 | 293 | 376 | 43.80% | 39.17% | 43.12% | 6.20 pp | -83 | 41 | -2.02 |
| BTC Hourly | rf | RandomForest | 847 | 377 | 470 | 44.51% | 43.33% | 43.96% | 5.49 pp | -93 | 45 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 495 | 197 | 298 | 39.80% | 37.50% | 40.42% | 10.20 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 669 | 285 | 384 | 42.60% | 40.00% | 43.75% | 7.40 pp | -99 | 41 | -2.41 |
| BTC Market Hours Daily | xgb | XGBoost | 495 | 194 | 301 | 39.19% | 36.25% | 39.38% | 10.81 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 847 | 363 | 484 | 42.86% | 39.58% | 42.29% | 7.14 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 847 | 356 | 491 | 42.03% | 40.00% | 42.29% | 7.97 pp | -135 | 45 | -3.00 |
| BTC Daily | xgb | XGBoost | 679 | 269 | 410 | 39.62% | 34.17% | 39.58% | 10.38 pp | -141 | 41 | -3.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 847 | 400 | 447 | 47.23% | 47.50% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 847 | 398 | 449 | 46.99% | 44.58% | 46.88% | 3.01 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 847 | 382 | 465 | 45.10% | 43.75% | 44.58% | 4.90 pp | -83 | 45 | -1.84 |
| BTC Hourly | rf | RandomForest | 847 | 377 | 470 | 44.51% | 43.33% | 43.96% | 5.49 pp | -93 | 45 | -2.07 |
| BTC Hourly | lstm | LSTM | 847 | 363 | 484 | 42.86% | 39.58% | 42.29% | 7.14 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 847 | 356 | 491 | 42.03% | 40.00% | 42.29% | 7.97 pp | -135 | 45 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 669 | 326 | 343 | 48.73% | 46.67% | 49.58% | 1.27 pp | -17 | 41 | -0.41 |
| BTC Daily | transformer | Transformer | 669 | 323 | 346 | 48.28% | 45.83% | 49.58% | 1.72 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 669 | 313 | 356 | 46.79% | 43.33% | 49.38% | 3.21 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 669 | 293 | 376 | 43.80% | 39.17% | 43.12% | 6.20 pp | -83 | 41 | -2.02 |
| BTC Daily | rf | RandomForest | 669 | 285 | 384 | 42.60% | 40.00% | 43.75% | 7.40 pp | -99 | 41 | -2.41 |
| BTC Daily | xgb | XGBoost | 679 | 269 | 410 | 39.62% | 34.17% | 39.58% | 10.38 pp | -141 | 41 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 441 | 215 | 226 | 48.75% | 44.58% | 48.75% | 1.25 pp | -11 | 43 | -0.26 |
| BTC Market Hours | nn | NN | 441 | 208 | 233 | 47.17% | 48.75% | 47.17% | 2.83 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 441 | 202 | 239 | 45.80% | 40.83% | 45.80% | 4.20 pp | -37 | 43 | -0.86 |
| BTC Market Hours | lstm | LSTM | 441 | 189 | 252 | 42.86% | 41.67% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 441 | 189 | 252 | 42.86% | 43.33% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 441 | 177 | 264 | 40.14% | 38.75% | 40.14% | 9.86 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 495 | 228 | 267 | 46.06% | 46.67% | 46.46% | 3.94 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | nn | NN | 495 | 227 | 268 | 45.86% | 44.17% | 46.46% | 4.14 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 495 | 224 | 271 | 45.25% | 45.00% | 45.21% | 4.75 pp | -47 | 43 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 495 | 204 | 291 | 41.21% | 41.25% | 41.46% | 8.79 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 495 | 197 | 298 | 39.80% | 37.50% | 40.42% | 10.20 pp | -101 | 43 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 495 | 194 | 301 | 39.19% | 36.25% | 39.38% | 10.81 pp | -107 | 43 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
