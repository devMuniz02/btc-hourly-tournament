# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T13:15:39.190195+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1227 | 939 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1103 | 738 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 12:00:00+00:00 | 772 | 500 | 271 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 12:00:00+00:00 | 774 | 554 | 218 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 149 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 149 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 36 | 113 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 36 | 113 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 500 | 242 | 258 | 48.40% | 45.42% | 48.33% | 1.60 pp | -16 | 48 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 500 | 237 | 263 | 47.40% | 51.25% | 48.12% | 2.60 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 500 | 236 | 264 | 47.20% | 44.58% | 47.71% | 2.80 pp | -28 | 48 | -0.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 728 | 351 | 377 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 554 | 259 | 295 | 46.75% | 49.17% | 47.71% | 3.25 pp | -36 | 48 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 905 | 434 | 471 | 47.96% | 51.67% | 48.54% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Daily | transformer | Transformer | 728 | 347 | 381 | 47.66% | 46.67% | 49.79% | 2.34 pp | -34 | 43 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 554 | 256 | 298 | 46.21% | 49.58% | 46.67% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 554 | 256 | 298 | 46.21% | 45.00% | 47.50% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Hourly | transformer | Transformer | 905 | 428 | 477 | 47.29% | 47.92% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 728 | 337 | 391 | 46.29% | 44.58% | 47.08% | 3.71 pp | -54 | 43 | -1.26 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 500 | 217 | 283 | 43.40% | 42.08% | 43.33% | 6.60 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 500 | 216 | 284 | 43.20% | 44.58% | 43.54% | 6.80 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 500 | 206 | 294 | 41.20% | 42.08% | 41.46% | 8.80 pp | -88 | 48 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 554 | 231 | 323 | 41.70% | 42.50% | 40.83% | 8.30 pp | -92 | 48 | -1.92 |
| BTC Hourly | nn | NN | 905 | 402 | 503 | 44.42% | 44.17% | 42.08% | 5.58 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 905 | 401 | 504 | 44.31% | 43.75% | 43.75% | 5.69 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 554 | 224 | 330 | 40.43% | 38.75% | 40.62% | 9.57 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 554 | 222 | 332 | 40.07% | 41.25% | 39.17% | 9.93 pp | -110 | 48 | -2.29 |
| BTC Daily | lstm | LSTM | 728 | 314 | 414 | 43.13% | 37.08% | 41.67% | 6.87 pp | -100 | 43 | -2.33 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 728 | 311 | 417 | 42.72% | 40.83% | 43.54% | 7.28 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 905 | 387 | 518 | 42.76% | 39.58% | 41.88% | 7.24 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 905 | 379 | 526 | 41.88% | 40.83% | 40.83% | 8.12 pp | -147 | 48 | -3.06 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 738 | 293 | 445 | 39.70% | 37.08% | 38.33% | 10.30 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 905 | 434 | 471 | 47.96% | 51.67% | 48.54% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Hourly | transformer | Transformer | 905 | 428 | 477 | 47.29% | 47.92% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Hourly | nn | NN | 905 | 402 | 503 | 44.42% | 44.17% | 42.08% | 5.58 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 905 | 401 | 504 | 44.31% | 43.75% | 43.75% | 5.69 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 905 | 387 | 518 | 42.76% | 39.58% | 41.88% | 7.24 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 905 | 379 | 526 | 41.88% | 40.83% | 40.83% | 8.12 pp | -147 | 48 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 728 | 351 | 377 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 728 | 347 | 381 | 47.66% | 46.67% | 49.79% | 2.34 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 728 | 337 | 391 | 46.29% | 44.58% | 47.08% | 3.71 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 728 | 314 | 414 | 43.13% | 37.08% | 41.67% | 6.87 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 728 | 311 | 417 | 42.72% | 40.83% | 43.54% | 7.28 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 738 | 293 | 445 | 39.70% | 37.08% | 38.33% | 10.30 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 500 | 242 | 258 | 48.40% | 45.42% | 48.33% | 1.60 pp | -16 | 48 | -0.33 |
| BTC Market Hours | nn | NN | 500 | 237 | 263 | 47.40% | 51.25% | 48.12% | 2.60 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 500 | 236 | 264 | 47.20% | 44.58% | 47.71% | 2.80 pp | -28 | 48 | -0.58 |
| BTC Market Hours | lstm | LSTM | 500 | 217 | 283 | 43.40% | 42.08% | 43.33% | 6.60 pp | -66 | 48 | -1.38 |
| BTC Market Hours | rf | RandomForest | 500 | 216 | 284 | 43.20% | 44.58% | 43.54% | 6.80 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 500 | 206 | 294 | 41.20% | 42.08% | 41.46% | 8.80 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 554 | 259 | 295 | 46.75% | 49.17% | 47.71% | 3.25 pp | -36 | 48 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 554 | 256 | 298 | 46.21% | 49.58% | 46.67% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 554 | 256 | 298 | 46.21% | 45.00% | 47.50% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 554 | 231 | 323 | 41.70% | 42.50% | 40.83% | 8.30 pp | -92 | 48 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 554 | 224 | 330 | 40.43% | 38.75% | 40.62% | 9.57 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 554 | 222 | 332 | 40.07% | 41.25% | 39.17% | 9.93 pp | -110 | 48 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
