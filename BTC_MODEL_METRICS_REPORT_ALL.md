# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T13:35:23.259774+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 12:00:00+00:00 | 773 | 501 | 271 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 501 | 242 | 259 | 48.30% | 45.42% | 48.12% | 1.70 pp | -17 | 48 | -0.35 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 501 | 238 | 263 | 47.50% | 51.25% | 48.12% | 2.50 pp | -25 | 48 | -0.52 |
| BTC Market Hours | transformer | Transformer | 501 | 237 | 264 | 47.31% | 45.00% | 47.92% | 2.69 pp | -27 | 48 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 729 | 352 | 377 | 48.29% | 46.25% | 47.92% | 1.71 pp | -25 | 43 | -0.58 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 554 | 259 | 295 | 46.75% | 49.17% | 47.71% | 3.25 pp | -36 | 48 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 906 | 434 | 472 | 47.90% | 51.25% | 48.33% | 2.10 pp | -38 | 48 | -0.79 |
| BTC Daily | transformer | Transformer | 729 | 347 | 382 | 47.60% | 46.67% | 49.58% | 2.40 pp | -35 | 43 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 554 | 256 | 298 | 46.21% | 49.58% | 46.67% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 554 | 256 | 298 | 46.21% | 45.00% | 47.50% | 3.79 pp | -42 | 48 | -0.88 |
| BTC Hourly | transformer | Transformer | 906 | 428 | 478 | 47.24% | 47.50% | 46.88% | 2.76 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 729 | 337 | 392 | 46.23% | 44.58% | 47.08% | 3.77 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 501 | 217 | 284 | 43.31% | 42.08% | 43.33% | 6.69 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 501 | 216 | 285 | 43.11% | 44.58% | 43.54% | 6.89 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 501 | 206 | 295 | 41.12% | 42.08% | 41.46% | 8.88 pp | -89 | 48 | -1.85 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 554 | 231 | 323 | 41.70% | 42.50% | 40.83% | 8.30 pp | -92 | 48 | -1.92 |
| BTC Hourly | nn | NN | 906 | 402 | 504 | 44.37% | 43.75% | 41.88% | 5.63 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 906 | 401 | 505 | 44.26% | 43.33% | 43.75% | 5.74 pp | -104 | 48 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 554 | 224 | 330 | 40.43% | 38.75% | 40.62% | 9.57 pp | -106 | 48 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 554 | 222 | 332 | 40.07% | 41.25% | 39.17% | 9.93 pp | -110 | 48 | -2.29 |
| BTC Daily | lstm | LSTM | 729 | 314 | 415 | 43.07% | 36.67% | 41.46% | 6.93 pp | -101 | 43 | -2.35 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 729 | 311 | 418 | 42.66% | 40.83% | 43.33% | 7.34 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 906 | 387 | 519 | 42.72% | 39.17% | 41.88% | 7.28 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 906 | 379 | 527 | 41.83% | 40.42% | 40.83% | 8.17 pp | -148 | 48 | -3.08 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 739 | 293 | 446 | 39.65% | 36.67% | 38.12% | 10.35 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

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
