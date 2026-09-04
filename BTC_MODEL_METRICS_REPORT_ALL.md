# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T15:03:37.086319+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1105 | 740 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 14:00:00+00:00 | 776 | 502 | 273 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 14:00:00+00:00 | 777 | 555 | 220 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 502 | 242 | 260 | 48.21% | 45.00% | 48.12% | 1.79 pp | -18 | 48 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 502 | 238 | 264 | 47.41% | 50.83% | 48.12% | 2.59 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 502 | 238 | 264 | 47.41% | 45.42% | 48.12% | 2.59 pp | -26 | 48 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 730 | 352 | 378 | 48.22% | 46.25% | 47.92% | 1.78 pp | -26 | 43 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 555 | 260 | 295 | 46.85% | 49.58% | 47.92% | 3.15 pp | -35 | 48 | -0.73 |
| BTC Daily | transformer | Transformer | 730 | 348 | 382 | 47.67% | 47.08% | 49.79% | 2.33 pp | -34 | 43 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 907 | 434 | 473 | 47.85% | 50.83% | 48.33% | 2.15 pp | -39 | 48 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 555 | 256 | 299 | 46.13% | 49.58% | 46.46% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 555 | 256 | 299 | 46.13% | 45.00% | 47.29% | 3.87 pp | -43 | 48 | -0.90 |
| BTC Hourly | transformer | Transformer | 907 | 429 | 478 | 47.30% | 47.92% | 46.88% | 2.70 pp | -49 | 48 | -1.02 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 730 | 337 | 393 | 46.16% | 44.58% | 46.88% | 3.84 pp | -56 | 43 | -1.30 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 502 | 217 | 285 | 43.23% | 42.08% | 43.33% | 6.77 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 502 | 216 | 286 | 43.03% | 44.58% | 43.33% | 6.97 pp | -70 | 48 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 502 | 206 | 296 | 41.04% | 42.08% | 41.46% | 8.96 pp | -90 | 48 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 555 | 231 | 324 | 41.62% | 42.50% | 40.62% | 8.38 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 907 | 403 | 504 | 44.43% | 43.75% | 42.08% | 5.57 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 907 | 402 | 505 | 44.32% | 43.75% | 43.75% | 5.68 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 555 | 225 | 330 | 40.54% | 38.75% | 40.62% | 9.46 pp | -105 | 48 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 555 | 222 | 333 | 40.00% | 41.25% | 38.96% | 10.00 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 730 | 314 | 416 | 43.01% | 36.67% | 41.46% | 6.99 pp | -102 | 43 | -2.37 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 730 | 311 | 419 | 42.60% | 40.83% | 43.33% | 7.40 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 907 | 388 | 519 | 42.78% | 39.58% | 42.08% | 7.22 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 907 | 380 | 527 | 41.90% | 40.83% | 40.83% | 8.10 pp | -147 | 48 | -3.06 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 740 | 293 | 447 | 39.59% | 36.25% | 38.12% | 10.41 pp | -154 | 43 | -3.58 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 730 | 352 | 378 | 48.22% | 46.25% | 47.92% | 1.78 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 730 | 348 | 382 | 47.67% | 47.08% | 49.79% | 2.33 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 730 | 337 | 393 | 46.16% | 44.58% | 46.88% | 3.84 pp | -56 | 43 | -1.30 |
| BTC Daily | lstm | LSTM | 730 | 314 | 416 | 43.01% | 36.67% | 41.46% | 6.99 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 730 | 311 | 419 | 42.60% | 40.83% | 43.33% | 7.40 pp | -108 | 43 | -2.51 |
| BTC Daily | xgb | XGBoost | 740 | 293 | 447 | 39.59% | 36.25% | 38.12% | 10.41 pp | -154 | 43 | -3.58 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 502 | 242 | 260 | 48.21% | 45.00% | 48.12% | 1.79 pp | -18 | 48 | -0.38 |
| BTC Market Hours | nn | NN | 502 | 238 | 264 | 47.41% | 50.83% | 48.12% | 2.59 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 502 | 238 | 264 | 47.41% | 45.42% | 48.12% | 2.59 pp | -26 | 48 | -0.54 |
| BTC Market Hours | lstm | LSTM | 502 | 217 | 285 | 43.23% | 42.08% | 43.33% | 6.77 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 502 | 216 | 286 | 43.03% | 44.58% | 43.33% | 6.97 pp | -70 | 48 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 502 | 206 | 296 | 41.04% | 42.08% | 41.46% | 8.96 pp | -90 | 48 | -1.88 |

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
