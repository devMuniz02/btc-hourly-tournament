# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T14:37:19.512909+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1196 | 908 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1072 | 707 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 13:00:00+00:00 | 716 | 469 | 246 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 13:00:00+00:00 | 718 | 523 | 193 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 119 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T17:00:00+00:00 | 120 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 469 | 227 | 242 | 48.40% | 43.75% | 48.40% | 1.60 pp | -15 | 45 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 697 | 341 | 356 | 48.92% | 46.67% | 49.38% | 1.08 pp | -15 | 42 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 469 | 220 | 249 | 46.91% | 47.50% | 46.91% | 3.09 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 469 | 219 | 250 | 46.70% | 41.25% | 46.70% | 3.30 pp | -31 | 45 | -0.69 |
| BTC Daily | transformer | Transformer | 697 | 334 | 363 | 47.92% | 46.25% | 49.17% | 2.08 pp | -29 | 42 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 523 | 240 | 283 | 45.89% | 46.67% | 46.46% | 4.11 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 523 | 240 | 283 | 45.89% | 47.08% | 46.67% | 4.11 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 874 | 414 | 460 | 47.37% | 48.33% | 47.71% | 2.63 pp | -46 | 46 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 874 | 413 | 461 | 47.25% | 48.75% | 47.71% | 2.75 pp | -48 | 46 | -1.04 |
| BTC Market Hours Daily | nn | NN | 523 | 238 | 285 | 45.51% | 42.50% | 46.04% | 4.49 pp | -47 | 45 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 697 | 324 | 373 | 46.48% | 43.33% | 48.54% | 3.52 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| BTC Market Hours | lstm | LSTM | 469 | 201 | 268 | 42.86% | 40.83% | 42.86% | 7.14 pp | -67 | 45 | -1.49 |
| BTC Market Hours | rf | RandomForest | 469 | 201 | 268 | 42.86% | 42.50% | 42.86% | 7.14 pp | -67 | 45 | -1.49 |
| Consolidated Hourly | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 874 | 393 | 481 | 44.97% | 46.25% | 43.96% | 5.03 pp | -88 | 46 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 469 | 190 | 279 | 40.51% | 39.58% | 40.51% | 9.49 pp | -89 | 45 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 523 | 216 | 307 | 41.30% | 41.25% | 41.25% | 8.70 pp | -91 | 45 | -2.02 |
| BTC Hourly | rf | RandomForest | 874 | 390 | 484 | 44.62% | 45.00% | 44.17% | 5.38 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 697 | 304 | 393 | 43.62% | 38.75% | 42.29% | 6.38 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 697 | 300 | 397 | 43.04% | 41.25% | 43.33% | 6.96 pp | -97 | 42 | -2.31 |
| BTC Market Hours Daily | lstm | LSTM | 523 | 209 | 314 | 39.96% | 37.08% | 40.83% | 10.04 pp | -105 | 45 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 523 | 207 | 316 | 39.58% | 37.50% | 38.96% | 10.42 pp | -109 | 45 | -2.42 |
| BTC Hourly | lstm | LSTM | 874 | 372 | 502 | 42.56% | 38.75% | 41.67% | 7.44 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 874 | 370 | 504 | 42.33% | 41.67% | 43.12% | 7.67 pp | -134 | 46 | -2.91 |
| BTC Daily | xgb | XGBoost | 707 | 281 | 426 | 39.75% | 36.67% | 39.58% | 10.25 pp | -145 | 42 | -3.45 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 874 | 414 | 460 | 47.37% | 48.33% | 47.71% | 2.63 pp | -46 | 46 | -1.00 |
| BTC Hourly | transformer | Transformer | 874 | 413 | 461 | 47.25% | 48.75% | 47.71% | 2.75 pp | -48 | 46 | -1.04 |
| BTC Hourly | nn | NN | 874 | 393 | 481 | 44.97% | 46.25% | 43.96% | 5.03 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 874 | 390 | 484 | 44.62% | 45.00% | 44.17% | 5.38 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 874 | 372 | 502 | 42.56% | 38.75% | 41.67% | 7.44 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 874 | 370 | 504 | 42.33% | 41.67% | 43.12% | 7.67 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 697 | 341 | 356 | 48.92% | 46.67% | 49.38% | 1.08 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 697 | 334 | 363 | 47.92% | 46.25% | 49.17% | 2.08 pp | -29 | 42 | -0.69 |
| BTC Daily | nn | NN | 697 | 324 | 373 | 46.48% | 43.33% | 48.54% | 3.52 pp | -49 | 42 | -1.17 |
| BTC Daily | lstm | LSTM | 697 | 304 | 393 | 43.62% | 38.75% | 42.29% | 6.38 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 697 | 300 | 397 | 43.04% | 41.25% | 43.33% | 6.96 pp | -97 | 42 | -2.31 |
| BTC Daily | xgb | XGBoost | 707 | 281 | 426 | 39.75% | 36.67% | 39.58% | 10.25 pp | -145 | 42 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 469 | 227 | 242 | 48.40% | 43.75% | 48.40% | 1.60 pp | -15 | 45 | -0.33 |
| BTC Market Hours | nn | NN | 469 | 220 | 249 | 46.91% | 47.50% | 46.91% | 3.09 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 469 | 219 | 250 | 46.70% | 41.25% | 46.70% | 3.30 pp | -31 | 45 | -0.69 |
| BTC Market Hours | lstm | LSTM | 469 | 201 | 268 | 42.86% | 40.83% | 42.86% | 7.14 pp | -67 | 45 | -1.49 |
| BTC Market Hours | rf | RandomForest | 469 | 201 | 268 | 42.86% | 42.50% | 42.86% | 7.14 pp | -67 | 45 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 469 | 190 | 279 | 40.51% | 39.58% | 40.51% | 9.49 pp | -89 | 45 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 523 | 240 | 283 | 45.89% | 46.67% | 46.46% | 4.11 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 523 | 240 | 283 | 45.89% | 47.08% | 46.67% | 4.11 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | nn | NN | 523 | 238 | 285 | 45.51% | 42.50% | 46.04% | 4.49 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | rf | RandomForest | 523 | 216 | 307 | 41.30% | 41.25% | 41.25% | 8.70 pp | -91 | 45 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 523 | 209 | 314 | 39.96% | 37.08% | 40.83% | 10.04 pp | -105 | 45 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 523 | 207 | 316 | 39.58% | 37.50% | 38.96% | 10.42 pp | -109 | 45 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 60 | 59 | 50.42% | 50.42% | 50.42% | 0.42 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 52 | 67 | 43.70% | 43.70% | 43.70% | 6.30 pp | -15 | 10 | -1.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 21 | 6 | 15 | 28.57% | 28.57% | 28.57% | 21.43 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 21 | 4 | 17 | 19.05% | 19.05% | 19.05% | 30.95 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
