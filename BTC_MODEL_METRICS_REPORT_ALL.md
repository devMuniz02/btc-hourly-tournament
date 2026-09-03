# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T23:47:47.724618+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1218 | 930 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1094 | 729 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 22:00:00+00:00 | 760 | 491 | 268 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 22:00:00+00:00 | 762 | 545 | 215 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 491 | 236 | 255 | 48.07% | 44.17% | 48.12% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 719 | 348 | 371 | 48.40% | 47.08% | 48.33% | 1.60 pp | -23 | 43 | -0.53 |
| BTC Market Hours | nn | NN | 491 | 231 | 260 | 47.05% | 49.58% | 47.71% | 2.95 pp | -29 | 47 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 491 | 229 | 262 | 46.64% | 42.92% | 47.29% | 3.36 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 719 | 343 | 376 | 47.71% | 45.83% | 50.00% | 2.29 pp | -33 | 43 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 896 | 429 | 467 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 545 | 253 | 292 | 46.42% | 49.17% | 47.50% | 3.58 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 545 | 250 | 295 | 45.87% | 47.92% | 47.08% | 4.13 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 545 | 250 | 295 | 45.87% | 44.17% | 46.88% | 4.13 pp | -45 | 47 | -0.96 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 896 | 424 | 472 | 47.32% | 48.33% | 46.88% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Daily | nn | NN | 719 | 333 | 386 | 46.31% | 43.75% | 47.92% | 3.69 pp | -53 | 43 | -1.23 |
| BTC Market Hours | lstm | LSTM | 491 | 212 | 279 | 43.18% | 41.25% | 43.33% | 6.82 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 491 | 212 | 279 | 43.18% | 43.33% | 43.54% | 6.82 pp | -67 | 47 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 491 | 201 | 290 | 40.94% | 40.00% | 40.83% | 9.06 pp | -89 | 47 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 545 | 226 | 319 | 41.47% | 41.67% | 41.25% | 8.53 pp | -93 | 47 | -1.98 |
| BTC Hourly | nn | NN | 896 | 399 | 497 | 44.53% | 44.58% | 42.50% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 896 | 399 | 497 | 44.53% | 45.42% | 44.17% | 5.47 pp | -98 | 47 | -2.09 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 545 | 219 | 326 | 40.18% | 37.92% | 40.83% | 9.82 pp | -107 | 47 | -2.28 |
| BTC Daily | lstm | LSTM | 719 | 310 | 409 | 43.12% | 37.50% | 41.88% | 6.88 pp | -99 | 43 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 545 | 218 | 327 | 40.00% | 40.42% | 39.58% | 10.00 pp | -109 | 47 | -2.32 |
| BTC Daily | rf | RandomForest | 719 | 306 | 413 | 42.56% | 40.00% | 43.33% | 7.44 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 896 | 383 | 513 | 42.75% | 39.17% | 42.29% | 7.25 pp | -130 | 47 | -2.77 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 896 | 377 | 519 | 42.08% | 42.92% | 41.88% | 7.92 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 729 | 289 | 440 | 39.64% | 35.83% | 38.54% | 10.36 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 896 | 429 | 467 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| BTC Hourly | transformer | Transformer | 896 | 424 | 472 | 47.32% | 48.33% | 46.88% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 896 | 399 | 497 | 44.53% | 44.58% | 42.50% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 896 | 399 | 497 | 44.53% | 45.42% | 44.17% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 896 | 383 | 513 | 42.75% | 39.17% | 42.29% | 7.25 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 896 | 377 | 519 | 42.08% | 42.92% | 41.88% | 7.92 pp | -142 | 47 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 719 | 348 | 371 | 48.40% | 47.08% | 48.33% | 1.60 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 719 | 343 | 376 | 47.71% | 45.83% | 50.00% | 2.29 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 719 | 333 | 386 | 46.31% | 43.75% | 47.92% | 3.69 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 719 | 310 | 409 | 43.12% | 37.50% | 41.88% | 6.88 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 719 | 306 | 413 | 42.56% | 40.00% | 43.33% | 7.44 pp | -107 | 43 | -2.49 |
| BTC Daily | xgb | XGBoost | 729 | 289 | 440 | 39.64% | 35.83% | 38.54% | 10.36 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 491 | 236 | 255 | 48.07% | 44.17% | 48.12% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 491 | 231 | 260 | 47.05% | 49.58% | 47.71% | 2.95 pp | -29 | 47 | -0.62 |
| BTC Market Hours | transformer | Transformer | 491 | 229 | 262 | 46.64% | 42.92% | 47.29% | 3.36 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 491 | 212 | 279 | 43.18% | 41.25% | 43.33% | 6.82 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 491 | 212 | 279 | 43.18% | 43.33% | 43.54% | 6.82 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 491 | 201 | 290 | 40.94% | 40.00% | 40.83% | 9.06 pp | -89 | 47 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 545 | 253 | 292 | 46.42% | 49.17% | 47.50% | 3.58 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 545 | 250 | 295 | 45.87% | 47.92% | 47.08% | 4.13 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 545 | 250 | 295 | 45.87% | 44.17% | 46.88% | 4.13 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 545 | 226 | 319 | 41.47% | 41.67% | 41.25% | 8.53 pp | -93 | 47 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 545 | 219 | 326 | 40.18% | 37.92% | 40.83% | 9.82 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 545 | 218 | 327 | 40.00% | 40.42% | 39.58% | 10.00 pp | -109 | 47 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
