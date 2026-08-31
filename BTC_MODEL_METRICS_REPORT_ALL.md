# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T11:53:56.117575+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1162 | 874 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1037 | 672 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 653 | 434 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 655 | 488 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 87 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 87 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 3 | 84 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 13:00:00+00:00 | 87 | 3 | 84 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 434 | 213 | 221 | 49.08% | 45.42% | 49.08% | 0.92 pp | -8 | 43 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 662 | 324 | 338 | 48.94% | 47.08% | 50.00% | 1.06 pp | -14 | 40 | -0.35 |
| BTC Daily | transformer | Transformer | 662 | 319 | 343 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Market Hours | nn | NN | 434 | 203 | 231 | 46.77% | 48.75% | 46.77% | 3.23 pp | -28 | 43 | -0.65 |
| Consolidated Hourly | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 434 | 199 | 235 | 45.85% | 41.25% | 45.85% | 4.15 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 488 | 225 | 263 | 46.11% | 47.08% | 46.25% | 3.89 pp | -38 | 43 | -0.88 |
| BTC Hourly | transformer | Transformer | 840 | 399 | 441 | 47.50% | 48.33% | 47.08% | 2.50 pp | -42 | 45 | -0.93 |
| BTC Daily | nn | NN | 662 | 311 | 351 | 46.98% | 43.33% | 49.58% | 3.02 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 488 | 222 | 266 | 45.49% | 43.33% | 45.83% | 4.51 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 488 | 221 | 267 | 45.29% | 45.00% | 45.42% | 4.71 pp | -46 | 43 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 840 | 393 | 447 | 46.79% | 43.33% | 46.46% | 3.21 pp | -54 | 45 | -1.20 |
| BTC Market Hours | lstm | LSTM | 434 | 187 | 247 | 43.09% | 42.92% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 434 | 187 | 247 | 43.09% | 43.33% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| Consolidated Hourly | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 840 | 380 | 460 | 45.24% | 44.17% | 44.79% | 4.76 pp | -80 | 45 | -1.78 |
| BTC Market Hours Daily | rf | RandomForest | 488 | 202 | 286 | 41.39% | 42.08% | 41.67% | 8.61 pp | -84 | 43 | -1.95 |
| BTC Hourly | rf | RandomForest | 840 | 375 | 465 | 44.64% | 43.75% | 44.17% | 5.36 pp | -90 | 45 | -2.00 |
| BTC Daily | lstm | LSTM | 662 | 291 | 371 | 43.96% | 39.58% | 43.54% | 6.04 pp | -80 | 40 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 434 | 174 | 260 | 40.09% | 38.33% | 40.09% | 9.91 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 488 | 196 | 292 | 40.16% | 39.17% | 40.42% | 9.84 pp | -96 | 43 | -2.23 |
| BTC Daily | rf | RandomForest | 662 | 284 | 378 | 42.90% | 41.67% | 44.17% | 7.10 pp | -94 | 40 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 488 | 190 | 298 | 38.93% | 35.83% | 39.17% | 11.07 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 840 | 361 | 479 | 42.98% | 40.00% | 42.50% | 7.02 pp | -118 | 45 | -2.62 |
| BTC Hourly | xgb | XGBoost | 840 | 355 | 485 | 42.26% | 40.00% | 42.71% | 7.74 pp | -130 | 45 | -2.89 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 672 | 267 | 405 | 39.73% | 33.75% | 40.00% | 10.27 pp | -138 | 40 | -3.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 840 | 399 | 441 | 47.50% | 48.33% | 47.08% | 2.50 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 840 | 393 | 447 | 46.79% | 43.33% | 46.46% | 3.21 pp | -54 | 45 | -1.20 |
| BTC Hourly | nn | NN | 840 | 380 | 460 | 45.24% | 44.17% | 44.79% | 4.76 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 840 | 375 | 465 | 44.64% | 43.75% | 44.17% | 5.36 pp | -90 | 45 | -2.00 |
| BTC Hourly | lstm | LSTM | 840 | 361 | 479 | 42.98% | 40.00% | 42.50% | 7.02 pp | -118 | 45 | -2.62 |
| BTC Hourly | xgb | XGBoost | 840 | 355 | 485 | 42.26% | 40.00% | 42.71% | 7.74 pp | -130 | 45 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 662 | 324 | 338 | 48.94% | 47.08% | 50.00% | 1.06 pp | -14 | 40 | -0.35 |
| BTC Daily | transformer | Transformer | 662 | 319 | 343 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Daily | nn | NN | 662 | 311 | 351 | 46.98% | 43.33% | 49.58% | 3.02 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 662 | 291 | 371 | 43.96% | 39.58% | 43.54% | 6.04 pp | -80 | 40 | -2.00 |
| BTC Daily | rf | RandomForest | 662 | 284 | 378 | 42.90% | 41.67% | 44.17% | 7.10 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 672 | 267 | 405 | 39.73% | 33.75% | 40.00% | 10.27 pp | -138 | 40 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 434 | 213 | 221 | 49.08% | 45.42% | 49.08% | 0.92 pp | -8 | 43 | -0.19 |
| BTC Market Hours | nn | NN | 434 | 203 | 231 | 46.77% | 48.75% | 46.77% | 3.23 pp | -28 | 43 | -0.65 |
| BTC Market Hours | transformer | Transformer | 434 | 199 | 235 | 45.85% | 41.25% | 45.85% | 4.15 pp | -36 | 43 | -0.84 |
| BTC Market Hours | lstm | LSTM | 434 | 187 | 247 | 43.09% | 42.92% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 434 | 187 | 247 | 43.09% | 43.33% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 434 | 174 | 260 | 40.09% | 38.33% | 40.09% | 9.91 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 488 | 225 | 263 | 46.11% | 47.08% | 46.25% | 3.89 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | nn | NN | 488 | 222 | 266 | 45.49% | 43.33% | 45.83% | 4.51 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 488 | 221 | 267 | 45.29% | 45.00% | 45.42% | 4.71 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 488 | 202 | 286 | 41.39% | 42.08% | 41.67% | 8.61 pp | -84 | 43 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 488 | 196 | 292 | 40.16% | 39.17% | 40.42% | 9.84 pp | -96 | 43 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 488 | 190 | 298 | 38.93% | 35.83% | 39.17% | 11.07 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 48 | 39 | 55.17% | 55.17% | 55.17% | 5.17 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 37 | 50 | 42.53% | 42.53% | 42.53% | 7.47 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
