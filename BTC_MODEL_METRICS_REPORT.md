# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T14:37:02.080375+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1245 | 957 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1121 | 756 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 13:00:00+00:00 | 804 | 518 | 285 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 13:00:00+00:00 | 806 | 572 | 232 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 164 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 164 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 164 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 165 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 518 | 252 | 266 | 48.65% | 46.25% | 48.75% | 1.35 pp | -14 | 49 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 518 | 250 | 268 | 48.26% | 47.92% | 48.75% | 1.74 pp | -18 | 49 | -0.37 |
| BTC Daily | mlp_sklearn | MLPClassifier | 746 | 363 | 383 | 48.66% | 47.92% | 48.96% | 1.34 pp | -20 | 44 | -0.45 |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 572 | 273 | 299 | 47.73% | 52.08% | 48.96% | 2.27 pp | -26 | 49 | -0.53 |
| BTC Market Hours | nn | NN | 518 | 245 | 273 | 47.30% | 50.42% | 48.54% | 2.70 pp | -28 | 49 | -0.57 |
| BTC Market Hours Daily | nn | NN | 572 | 266 | 306 | 46.50% | 46.25% | 47.71% | 3.50 pp | -40 | 49 | -0.82 |
| BTC Daily | transformer | Transformer | 746 | 355 | 391 | 47.59% | 45.42% | 49.38% | 2.41 pp | -36 | 44 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 923 | 441 | 482 | 47.78% | 49.58% | 47.29% | 2.22 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 572 | 263 | 309 | 45.98% | 49.58% | 46.46% | 4.02 pp | -46 | 49 | -0.94 |
| BTC Hourly | transformer | Transformer | 923 | 436 | 487 | 47.24% | 47.08% | 45.83% | 2.76 pp | -51 | 48 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 12 | -1.17 |
| BTC Daily | nn | NN | 746 | 346 | 400 | 46.38% | 43.75% | 47.08% | 3.62 pp | -54 | 44 | -1.23 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 518 | 226 | 292 | 43.63% | 43.33% | 44.17% | 6.37 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 518 | 224 | 294 | 43.24% | 45.42% | 43.75% | 6.76 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 164 | 73 | 91 | 44.51% | 44.51% | 44.51% | 5.49 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 164 | 73 | 91 | 44.51% | 44.51% | 44.51% | 5.49 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 518 | 214 | 304 | 41.31% | 43.33% | 42.08% | 8.69 pp | -90 | 49 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 572 | 239 | 333 | 41.78% | 42.92% | 41.04% | 8.22 pp | -94 | 49 | -1.92 |
| BTC Hourly | nn | NN | 923 | 411 | 512 | 44.53% | 43.33% | 42.50% | 5.47 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 923 | 411 | 512 | 44.53% | 43.75% | 43.96% | 5.47 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 572 | 234 | 338 | 40.91% | 40.83% | 41.04% | 9.09 pp | -104 | 49 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 12 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 572 | 228 | 344 | 39.86% | 41.25% | 39.17% | 10.14 pp | -116 | 49 | -2.37 |
| BTC Daily | lstm | LSTM | 746 | 319 | 427 | 42.76% | 36.25% | 40.83% | 7.24 pp | -108 | 44 | -2.45 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 746 | 315 | 431 | 42.23% | 38.75% | 42.29% | 7.77 pp | -116 | 44 | -2.64 |
| BTC Hourly | lstm | LSTM | 923 | 394 | 529 | 42.69% | 37.92% | 41.46% | 7.31 pp | -135 | 48 | -2.81 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 923 | 386 | 537 | 41.82% | 39.58% | 40.21% | 8.18 pp | -151 | 48 | -3.15 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 756 | 298 | 458 | 39.42% | 36.67% | 37.29% | 10.58 pp | -160 | 44 | -3.64 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 923 | 441 | 482 | 47.78% | 49.58% | 47.29% | 2.22 pp | -41 | 48 | -0.85 |
| BTC Hourly | transformer | Transformer | 923 | 436 | 487 | 47.24% | 47.08% | 45.83% | 2.76 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 923 | 411 | 512 | 44.53% | 43.33% | 42.50% | 5.47 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 923 | 411 | 512 | 44.53% | 43.75% | 43.96% | 5.47 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 923 | 394 | 529 | 42.69% | 37.92% | 41.46% | 7.31 pp | -135 | 48 | -2.81 |
| BTC Hourly | xgb | XGBoost | 923 | 386 | 537 | 41.82% | 39.58% | 40.21% | 8.18 pp | -151 | 48 | -3.15 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 746 | 363 | 383 | 48.66% | 47.92% | 48.96% | 1.34 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 746 | 355 | 391 | 47.59% | 45.42% | 49.38% | 2.41 pp | -36 | 44 | -0.82 |
| BTC Daily | nn | NN | 746 | 346 | 400 | 46.38% | 43.75% | 47.08% | 3.62 pp | -54 | 44 | -1.23 |
| BTC Daily | lstm | LSTM | 746 | 319 | 427 | 42.76% | 36.25% | 40.83% | 7.24 pp | -108 | 44 | -2.45 |
| BTC Daily | rf | RandomForest | 746 | 315 | 431 | 42.23% | 38.75% | 42.29% | 7.77 pp | -116 | 44 | -2.64 |
| BTC Daily | xgb | XGBoost | 756 | 298 | 458 | 39.42% | 36.67% | 37.29% | 10.58 pp | -160 | 44 | -3.64 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 518 | 252 | 266 | 48.65% | 46.25% | 48.75% | 1.35 pp | -14 | 49 | -0.29 |
| BTC Market Hours | transformer | Transformer | 518 | 250 | 268 | 48.26% | 47.92% | 48.75% | 1.74 pp | -18 | 49 | -0.37 |
| BTC Market Hours | nn | NN | 518 | 245 | 273 | 47.30% | 50.42% | 48.54% | 2.70 pp | -28 | 49 | -0.57 |
| BTC Market Hours | lstm | LSTM | 518 | 226 | 292 | 43.63% | 43.33% | 44.17% | 6.37 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 518 | 224 | 294 | 43.24% | 45.42% | 43.75% | 6.76 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 518 | 214 | 304 | 41.31% | 43.33% | 42.08% | 8.69 pp | -90 | 49 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 572 | 273 | 299 | 47.73% | 52.08% | 48.96% | 2.27 pp | -26 | 49 | -0.53 |
| BTC Market Hours Daily | nn | NN | 572 | 266 | 306 | 46.50% | 46.25% | 47.71% | 3.50 pp | -40 | 49 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 572 | 263 | 309 | 45.98% | 49.58% | 46.46% | 4.02 pp | -46 | 49 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 572 | 239 | 333 | 41.78% | 42.92% | 41.04% | 8.22 pp | -94 | 49 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 572 | 234 | 338 | 40.91% | 40.83% | 41.04% | 9.09 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 572 | 228 | 344 | 39.86% | 41.25% | 39.17% | 10.14 pp | -116 | 49 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 164 | 73 | 91 | 44.51% | 44.51% | 44.51% | 5.49 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 164 | 80 | 84 | 48.78% | 48.78% | 48.78% | 1.22 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 164 | 75 | 89 | 45.73% | 45.73% | 45.73% | 4.27 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | nn | NN | 164 | 74 | 90 | 45.12% | 45.12% | 45.12% | 4.88 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 164 | 73 | 91 | 44.51% | 44.51% | 44.51% | 5.49 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 164 | 69 | 95 | 42.07% | 42.07% | 42.07% | 7.93 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
