# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T12:49:12.971413+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1195 | 907 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1071 | 706 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 713 | 468 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 715 | 522 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 119 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 468 | 227 | 241 | 48.50% | 44.17% | 48.50% | 1.50 pp | -14 | 45 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 696 | 340 | 356 | 48.85% | 46.25% | 49.38% | 1.15 pp | -16 | 42 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | transformer | Transformer | 696 | 334 | 362 | 47.99% | 46.25% | 49.17% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Market Hours | nn | NN | 468 | 219 | 249 | 46.79% | 47.50% | 46.79% | 3.21 pp | -30 | 45 | -0.67 |
| BTC Market Hours | transformer | Transformer | 468 | 219 | 249 | 46.79% | 41.67% | 46.79% | 3.21 pp | -30 | 45 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 522 | 240 | 282 | 45.98% | 47.08% | 46.46% | 4.02 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 522 | 239 | 283 | 45.79% | 47.08% | 46.46% | 4.21 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 873 | 414 | 459 | 47.42% | 48.33% | 47.71% | 2.58 pp | -45 | 46 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 522 | 238 | 284 | 45.59% | 42.92% | 46.25% | 4.41 pp | -46 | 45 | -1.02 |
| BTC Hourly | transformer | Transformer | 873 | 412 | 461 | 47.19% | 48.33% | 47.50% | 2.81 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 696 | 323 | 373 | 46.41% | 42.92% | 48.54% | 3.59 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| BTC Market Hours | lstm | LSTM | 468 | 201 | 267 | 42.95% | 41.25% | 42.95% | 7.05 pp | -66 | 45 | -1.47 |
| BTC Market Hours | rf | RandomForest | 468 | 201 | 267 | 42.95% | 42.92% | 42.95% | 7.05 pp | -66 | 45 | -1.47 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |
| BTC Hourly | nn | NN | 873 | 393 | 480 | 45.02% | 46.25% | 43.96% | 4.98 pp | -87 | 46 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 468 | 190 | 278 | 40.60% | 39.58% | 40.60% | 9.40 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 522 | 216 | 306 | 41.38% | 41.25% | 41.46% | 8.62 pp | -90 | 45 | -2.00 |
| BTC Hourly | rf | RandomForest | 873 | 389 | 484 | 44.56% | 44.58% | 43.96% | 5.44 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 696 | 303 | 393 | 43.53% | 38.33% | 42.29% | 6.47 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 522 | 209 | 313 | 40.04% | 37.50% | 41.04% | 9.96 pp | -104 | 45 | -2.31 |
| BTC Daily | rf | RandomForest | 696 | 299 | 397 | 42.96% | 40.83% | 43.33% | 7.04 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 522 | 207 | 315 | 39.66% | 37.50% | 39.17% | 10.34 pp | -108 | 45 | -2.40 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 873 | 372 | 501 | 42.61% | 38.75% | 41.67% | 7.39 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 873 | 369 | 504 | 42.27% | 41.25% | 42.92% | 7.73 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 706 | 280 | 426 | 39.66% | 36.25% | 39.38% | 10.34 pp | -146 | 42 | -3.48 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 873 | 414 | 459 | 47.42% | 48.33% | 47.71% | 2.58 pp | -45 | 46 | -0.98 |
| BTC Hourly | transformer | Transformer | 873 | 412 | 461 | 47.19% | 48.33% | 47.50% | 2.81 pp | -49 | 46 | -1.07 |
| BTC Hourly | nn | NN | 873 | 393 | 480 | 45.02% | 46.25% | 43.96% | 4.98 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 873 | 389 | 484 | 44.56% | 44.58% | 43.96% | 5.44 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 873 | 372 | 501 | 42.61% | 38.75% | 41.67% | 7.39 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 873 | 369 | 504 | 42.27% | 41.25% | 42.92% | 7.73 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 696 | 340 | 356 | 48.85% | 46.25% | 49.38% | 1.15 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 696 | 334 | 362 | 47.99% | 46.25% | 49.17% | 2.01 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 696 | 323 | 373 | 46.41% | 42.92% | 48.54% | 3.59 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 696 | 303 | 393 | 43.53% | 38.33% | 42.29% | 6.47 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 696 | 299 | 397 | 42.96% | 40.83% | 43.33% | 7.04 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 706 | 280 | 426 | 39.66% | 36.25% | 39.38% | 10.34 pp | -146 | 42 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 468 | 227 | 241 | 48.50% | 44.17% | 48.50% | 1.50 pp | -14 | 45 | -0.31 |
| BTC Market Hours | nn | NN | 468 | 219 | 249 | 46.79% | 47.50% | 46.79% | 3.21 pp | -30 | 45 | -0.67 |
| BTC Market Hours | transformer | Transformer | 468 | 219 | 249 | 46.79% | 41.67% | 46.79% | 3.21 pp | -30 | 45 | -0.67 |
| BTC Market Hours | lstm | LSTM | 468 | 201 | 267 | 42.95% | 41.25% | 42.95% | 7.05 pp | -66 | 45 | -1.47 |
| BTC Market Hours | rf | RandomForest | 468 | 201 | 267 | 42.95% | 42.92% | 42.95% | 7.05 pp | -66 | 45 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 468 | 190 | 278 | 40.60% | 39.58% | 40.60% | 9.40 pp | -88 | 45 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 522 | 240 | 282 | 45.98% | 47.08% | 46.46% | 4.02 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 522 | 239 | 283 | 45.79% | 47.08% | 46.46% | 4.21 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 522 | 238 | 284 | 45.59% | 42.92% | 46.25% | 4.41 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 522 | 216 | 306 | 41.38% | 41.25% | 41.46% | 8.62 pp | -90 | 45 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 522 | 209 | 313 | 40.04% | 37.50% | 41.04% | 9.96 pp | -104 | 45 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 522 | 207 | 315 | 39.66% | 37.50% | 39.17% | 10.34 pp | -108 | 45 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
