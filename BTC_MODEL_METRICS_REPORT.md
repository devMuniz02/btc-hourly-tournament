# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T14:16:11.277145+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1228 | 940 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1104 | 739 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 774 | 501 | 272 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 776 | 555 | 219 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 150 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 501 | 242 | 259 | 48.30% | 45.42% | 48.12% | 1.70 pp | -17 | 48 | -0.35 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 501 | 238 | 263 | 47.50% | 51.25% | 48.12% | 2.50 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 501 | 237 | 264 | 47.31% | 45.00% | 47.92% | 2.69 pp | -27 | 48 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 729 | 352 | 377 | 48.29% | 46.25% | 47.92% | 1.71 pp | -25 | 43 | -0.58 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 555 | 260 | 295 | 46.85% | 49.58% | 47.92% | 3.15 pp | -35 | 48 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 906 | 434 | 472 | 47.90% | 51.25% | 48.33% | 2.10 pp | -38 | 48 | -0.79 |
| BTC Daily | transformer | Transformer | 729 | 347 | 382 | 47.60% | 46.67% | 49.58% | 2.40 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 555 | 256 | 299 | 46.13% | 49.58% | 46.46% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 555 | 256 | 299 | 46.13% | 45.00% | 47.29% | 3.87 pp | -43 | 48 | -0.90 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 906 | 428 | 478 | 47.24% | 47.50% | 46.88% | 2.76 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 729 | 337 | 392 | 46.23% | 44.58% | 47.08% | 3.77 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 501 | 217 | 284 | 43.31% | 42.08% | 43.33% | 6.69 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 501 | 216 | 285 | 43.11% | 44.58% | 43.54% | 6.89 pp | -69 | 48 | -1.44 |
| Consolidated Hourly | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 501 | 206 | 295 | 41.12% | 42.08% | 41.46% | 8.88 pp | -89 | 48 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 555 | 231 | 324 | 41.62% | 42.50% | 40.62% | 8.38 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 906 | 402 | 504 | 44.37% | 43.75% | 41.88% | 5.63 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 906 | 401 | 505 | 44.26% | 43.33% | 43.75% | 5.74 pp | -104 | 48 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 555 | 225 | 330 | 40.54% | 38.75% | 40.62% | 9.46 pp | -105 | 48 | -2.19 |
| Consolidated Hourly | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 555 | 222 | 333 | 40.00% | 41.25% | 38.96% | 10.00 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 729 | 314 | 415 | 43.07% | 36.67% | 41.46% | 6.93 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 729 | 311 | 418 | 42.66% | 40.83% | 43.33% | 7.34 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 906 | 387 | 519 | 42.72% | 39.17% | 41.88% | 7.28 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 906 | 379 | 527 | 41.83% | 40.42% | 40.83% | 8.17 pp | -148 | 48 | -3.08 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 739 | 293 | 446 | 39.65% | 36.67% | 38.12% | 10.35 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 906 | 434 | 472 | 47.90% | 51.25% | 48.33% | 2.10 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 906 | 428 | 478 | 47.24% | 47.50% | 46.88% | 2.76 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 906 | 402 | 504 | 44.37% | 43.75% | 41.88% | 5.63 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 906 | 401 | 505 | 44.26% | 43.33% | 43.75% | 5.74 pp | -104 | 48 | -2.17 |
| BTC Hourly | lstm | LSTM | 906 | 387 | 519 | 42.72% | 39.17% | 41.88% | 7.28 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 906 | 379 | 527 | 41.83% | 40.42% | 40.83% | 8.17 pp | -148 | 48 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 729 | 352 | 377 | 48.29% | 46.25% | 47.92% | 1.71 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 729 | 347 | 382 | 47.60% | 46.67% | 49.58% | 2.40 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 729 | 337 | 392 | 46.23% | 44.58% | 47.08% | 3.77 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 729 | 314 | 415 | 43.07% | 36.67% | 41.46% | 6.93 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 729 | 311 | 418 | 42.66% | 40.83% | 43.33% | 7.34 pp | -107 | 43 | -2.49 |
| BTC Daily | xgb | XGBoost | 739 | 293 | 446 | 39.65% | 36.67% | 38.12% | 10.35 pp | -153 | 43 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 501 | 242 | 259 | 48.30% | 45.42% | 48.12% | 1.70 pp | -17 | 48 | -0.35 |
| BTC Market Hours | nn | NN | 501 | 238 | 263 | 47.50% | 51.25% | 48.12% | 2.50 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 501 | 237 | 264 | 47.31% | 45.00% | 47.92% | 2.69 pp | -27 | 48 | -0.56 |
| BTC Market Hours | lstm | LSTM | 501 | 217 | 284 | 43.31% | 42.08% | 43.33% | 6.69 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 501 | 216 | 285 | 43.11% | 44.58% | 43.54% | 6.89 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 501 | 206 | 295 | 41.12% | 42.08% | 41.46% | 8.88 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 555 | 260 | 295 | 46.85% | 49.58% | 47.92% | 3.15 pp | -35 | 48 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 555 | 256 | 299 | 46.13% | 49.58% | 46.46% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 555 | 256 | 299 | 46.13% | 45.00% | 47.29% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 555 | 231 | 324 | 41.62% | 42.50% | 40.62% | 8.38 pp | -93 | 48 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 555 | 225 | 330 | 40.54% | 38.75% | 40.62% | 9.46 pp | -105 | 48 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 555 | 222 | 333 | 40.00% | 41.25% | 38.96% | 10.00 pp | -111 | 48 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
