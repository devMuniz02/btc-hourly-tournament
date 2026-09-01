# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T14:39:40.523737+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1180 | 892 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1056 | 691 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 687 | 453 | 233 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 689 | 507 | 180 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 681 | 334 | 347 | 49.05% | 47.50% | 49.58% | 0.95 pp | -13 | 41 | -0.32 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 453 | 219 | 234 | 48.34% | 44.17% | 48.34% | 1.66 pp | -15 | 44 | -0.34 |
| BTC Daily | transformer | Transformer | 681 | 329 | 352 | 48.31% | 46.25% | 49.58% | 1.69 pp | -23 | 41 | -0.56 |
| BTC Market Hours | nn | NN | 453 | 214 | 239 | 47.24% | 48.75% | 47.24% | 2.76 pp | -25 | 44 | -0.57 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 453 | 209 | 244 | 46.14% | 40.00% | 46.14% | 3.86 pp | -35 | 44 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 507 | 232 | 275 | 45.76% | 45.83% | 46.25% | 4.24 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 507 | 232 | 275 | 45.76% | 43.33% | 46.67% | 4.24 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 507 | 231 | 276 | 45.56% | 46.67% | 45.83% | 4.44 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 858 | 405 | 453 | 47.20% | 45.42% | 46.88% | 2.80 pp | -48 | 46 | -1.04 |
| BTC Daily | nn | NN | 681 | 319 | 362 | 46.84% | 43.33% | 49.17% | 3.16 pp | -43 | 41 | -1.05 |
| BTC Hourly | transformer | Transformer | 858 | 404 | 454 | 47.09% | 47.08% | 46.67% | 2.91 pp | -50 | 46 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| BTC Market Hours | rf | RandomForest | 453 | 196 | 257 | 43.27% | 43.33% | 43.27% | 6.73 pp | -61 | 44 | -1.39 |
| BTC Market Hours | lstm | LSTM | 453 | 192 | 261 | 42.38% | 40.00% | 42.38% | 7.62 pp | -69 | 44 | -1.57 |
| BTC Hourly | nn | NN | 858 | 387 | 471 | 45.10% | 45.42% | 44.38% | 4.90 pp | -84 | 46 | -1.83 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 507 | 211 | 296 | 41.62% | 41.25% | 41.67% | 8.38 pp | -85 | 44 | -1.93 |
| BTC Hourly | rf | RandomForest | 858 | 383 | 475 | 44.64% | 43.75% | 43.96% | 5.36 pp | -92 | 46 | -2.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 453 | 182 | 271 | 40.18% | 38.33% | 40.18% | 9.82 pp | -89 | 44 | -2.02 |
| BTC Daily | lstm | LSTM | 681 | 297 | 384 | 43.61% | 38.75% | 42.50% | 6.39 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 681 | 293 | 388 | 43.02% | 40.83% | 43.54% | 6.98 pp | -95 | 41 | -2.32 |
| BTC Market Hours Daily | lstm | LSTM | 507 | 202 | 305 | 39.84% | 37.50% | 40.62% | 10.16 pp | -103 | 44 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 507 | 199 | 308 | 39.25% | 35.83% | 38.75% | 10.75 pp | -109 | 44 | -2.48 |
| BTC Hourly | lstm | LSTM | 858 | 365 | 493 | 42.54% | 37.50% | 41.67% | 7.46 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 858 | 362 | 496 | 42.19% | 40.42% | 42.50% | 7.81 pp | -134 | 46 | -2.91 |
| BTC Daily | xgb | XGBoost | 691 | 275 | 416 | 39.80% | 35.00% | 39.58% | 10.20 pp | -141 | 41 | -3.44 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 858 | 405 | 453 | 47.20% | 45.42% | 46.88% | 2.80 pp | -48 | 46 | -1.04 |
| BTC Hourly | transformer | Transformer | 858 | 404 | 454 | 47.09% | 47.08% | 46.67% | 2.91 pp | -50 | 46 | -1.09 |
| BTC Hourly | nn | NN | 858 | 387 | 471 | 45.10% | 45.42% | 44.38% | 4.90 pp | -84 | 46 | -1.83 |
| BTC Hourly | rf | RandomForest | 858 | 383 | 475 | 44.64% | 43.75% | 43.96% | 5.36 pp | -92 | 46 | -2.00 |
| BTC Hourly | lstm | LSTM | 858 | 365 | 493 | 42.54% | 37.50% | 41.67% | 7.46 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 858 | 362 | 496 | 42.19% | 40.42% | 42.50% | 7.81 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 681 | 334 | 347 | 49.05% | 47.50% | 49.58% | 0.95 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 681 | 329 | 352 | 48.31% | 46.25% | 49.58% | 1.69 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 681 | 319 | 362 | 46.84% | 43.33% | 49.17% | 3.16 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 681 | 297 | 384 | 43.61% | 38.75% | 42.50% | 6.39 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 681 | 293 | 388 | 43.02% | 40.83% | 43.54% | 6.98 pp | -95 | 41 | -2.32 |
| BTC Daily | xgb | XGBoost | 691 | 275 | 416 | 39.80% | 35.00% | 39.58% | 10.20 pp | -141 | 41 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 453 | 219 | 234 | 48.34% | 44.17% | 48.34% | 1.66 pp | -15 | 44 | -0.34 |
| BTC Market Hours | nn | NN | 453 | 214 | 239 | 47.24% | 48.75% | 47.24% | 2.76 pp | -25 | 44 | -0.57 |
| BTC Market Hours | transformer | Transformer | 453 | 209 | 244 | 46.14% | 40.00% | 46.14% | 3.86 pp | -35 | 44 | -0.80 |
| BTC Market Hours | rf | RandomForest | 453 | 196 | 257 | 43.27% | 43.33% | 43.27% | 6.73 pp | -61 | 44 | -1.39 |
| BTC Market Hours | lstm | LSTM | 453 | 192 | 261 | 42.38% | 40.00% | 42.38% | 7.62 pp | -69 | 44 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 453 | 182 | 271 | 40.18% | 38.33% | 40.18% | 9.82 pp | -89 | 44 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 507 | 232 | 275 | 45.76% | 45.83% | 46.25% | 4.24 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 507 | 232 | 275 | 45.76% | 43.33% | 46.67% | 4.24 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 507 | 231 | 276 | 45.56% | 46.67% | 45.83% | 4.44 pp | -45 | 44 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 507 | 211 | 296 | 41.62% | 41.25% | 41.67% | 8.38 pp | -85 | 44 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 507 | 202 | 305 | 39.84% | 37.50% | 40.62% | 10.16 pp | -103 | 44 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 507 | 199 | 308 | 39.25% | 35.83% | 38.75% | 10.75 pp | -109 | 44 | -2.48 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
