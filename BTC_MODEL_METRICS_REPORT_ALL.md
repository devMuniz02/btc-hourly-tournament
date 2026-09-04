# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T14:53:11.117180+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1229 | 941 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1104 | 739 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 774 | 501 | 272 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 776 | 555 | 219 | 2 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 729 | 351 | 378 | 48.15% | 45.83% | 47.71% | 1.85 pp | -27 | 43 | -0.63 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 555 | 260 | 295 | 46.85% | 49.58% | 47.92% | 3.15 pp | -35 | 48 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 907 | 434 | 473 | 47.85% | 50.83% | 48.33% | 2.15 pp | -39 | 48 | -0.81 |
| BTC Daily | transformer | Transformer | 729 | 347 | 382 | 47.60% | 46.67% | 49.58% | 2.40 pp | -35 | 43 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 555 | 256 | 299 | 46.13% | 49.58% | 46.46% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 555 | 256 | 299 | 46.13% | 45.00% | 47.29% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Hourly | transformer | Transformer | 907 | 429 | 478 | 47.30% | 47.92% | 46.88% | 2.70 pp | -49 | 48 | -1.02 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 729 | 336 | 393 | 46.09% | 44.17% | 46.88% | 3.91 pp | -57 | 43 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 501 | 217 | 284 | 43.31% | 42.08% | 43.33% | 6.69 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 501 | 216 | 285 | 43.11% | 44.58% | 43.54% | 6.89 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 501 | 206 | 295 | 41.12% | 42.08% | 41.46% | 8.88 pp | -89 | 48 | -1.85 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 555 | 231 | 324 | 41.62% | 42.50% | 40.62% | 8.38 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 907 | 403 | 504 | 44.43% | 43.75% | 42.08% | 5.57 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 907 | 402 | 505 | 44.32% | 43.75% | 43.75% | 5.68 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 555 | 225 | 330 | 40.54% | 38.75% | 40.62% | 9.46 pp | -105 | 48 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 555 | 222 | 333 | 40.00% | 41.25% | 38.96% | 10.00 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 729 | 314 | 415 | 43.07% | 36.67% | 41.46% | 6.93 pp | -101 | 43 | -2.35 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 729 | 310 | 419 | 42.52% | 40.42% | 43.12% | 7.48 pp | -109 | 43 | -2.53 |
| BTC Hourly | lstm | LSTM | 907 | 388 | 519 | 42.78% | 39.58% | 42.08% | 7.22 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 907 | 380 | 527 | 41.90% | 40.83% | 40.83% | 8.10 pp | -147 | 48 | -3.06 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 739 | 292 | 447 | 39.51% | 36.25% | 37.92% | 10.49 pp | -155 | 43 | -3.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 907 | 434 | 473 | 47.85% | 50.83% | 48.33% | 2.15 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 907 | 429 | 478 | 47.30% | 47.92% | 46.88% | 2.70 pp | -49 | 48 | -1.02 |
| BTC Hourly | nn | NN | 907 | 403 | 504 | 44.43% | 43.75% | 42.08% | 5.57 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 907 | 402 | 505 | 44.32% | 43.75% | 43.75% | 5.68 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 907 | 388 | 519 | 42.78% | 39.58% | 42.08% | 7.22 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 907 | 380 | 527 | 41.90% | 40.83% | 40.83% | 8.10 pp | -147 | 48 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 729 | 351 | 378 | 48.15% | 45.83% | 47.71% | 1.85 pp | -27 | 43 | -0.63 |
| BTC Daily | transformer | Transformer | 729 | 347 | 382 | 47.60% | 46.67% | 49.58% | 2.40 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 729 | 336 | 393 | 46.09% | 44.17% | 46.88% | 3.91 pp | -57 | 43 | -1.33 |
| BTC Daily | lstm | LSTM | 729 | 314 | 415 | 43.07% | 36.67% | 41.46% | 6.93 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 729 | 310 | 419 | 42.52% | 40.42% | 43.12% | 7.48 pp | -109 | 43 | -2.53 |
| BTC Daily | xgb | XGBoost | 739 | 292 | 447 | 39.51% | 36.25% | 37.92% | 10.49 pp | -155 | 43 | -3.60 |

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
