# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T22:29:04.260143+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1093 | 728 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 758 | 490 | 267 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 21:00:00+00:00 | 760 | 544 | 214 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 490 | 236 | 254 | 48.16% | 44.17% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 718 | 347 | 371 | 48.33% | 46.67% | 48.33% | 1.67 pp | -24 | 43 | -0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| BTC Market Hours | nn | NN | 490 | 230 | 260 | 46.94% | 49.17% | 47.50% | 3.06 pp | -30 | 47 | -0.64 |
| BTC Market Hours | transformer | Transformer | 490 | 229 | 261 | 46.73% | 42.92% | 47.29% | 3.27 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 718 | 342 | 376 | 47.63% | 45.42% | 50.00% | 2.37 pp | -34 | 43 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 896 | 428 | 468 | 47.77% | 50.83% | 48.54% | 2.23 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 544 | 252 | 292 | 46.32% | 49.17% | 47.50% | 3.68 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 544 | 249 | 295 | 45.77% | 47.50% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 544 | 249 | 295 | 45.77% | 44.17% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 896 | 424 | 472 | 47.32% | 48.33% | 46.88% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Daily | nn | NN | 718 | 332 | 386 | 46.24% | 43.33% | 47.71% | 3.76 pp | -54 | 43 | -1.26 |
| BTC Market Hours | lstm | LSTM | 490 | 211 | 279 | 43.06% | 40.83% | 43.12% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 490 | 211 | 279 | 43.06% | 42.92% | 43.33% | 6.94 pp | -68 | 47 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 490 | 200 | 290 | 40.82% | 40.00% | 40.83% | 9.18 pp | -90 | 47 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 544 | 225 | 319 | 41.36% | 41.67% | 41.25% | 8.64 pp | -94 | 47 | -2.00 |
| BTC Hourly | nn | NN | 896 | 399 | 497 | 44.53% | 44.58% | 42.50% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 896 | 399 | 497 | 44.53% | 45.42% | 44.17% | 5.47 pp | -98 | 47 | -2.09 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 544 | 219 | 325 | 40.26% | 37.92% | 40.83% | 9.74 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 718 | 310 | 408 | 43.18% | 37.50% | 42.08% | 6.82 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 544 | 217 | 327 | 39.89% | 40.00% | 39.38% | 10.11 pp | -110 | 47 | -2.34 |
| BTC Daily | rf | RandomForest | 718 | 305 | 413 | 42.48% | 40.00% | 43.33% | 7.52 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 896 | 383 | 513 | 42.75% | 39.17% | 42.29% | 7.25 pp | -130 | 47 | -2.77 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 896 | 377 | 519 | 42.08% | 42.92% | 41.88% | 7.92 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 728 | 288 | 440 | 39.56% | 35.42% | 38.54% | 10.44 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 896 | 428 | 468 | 47.77% | 50.83% | 48.54% | 2.23 pp | -40 | 47 | -0.85 |
| BTC Hourly | transformer | Transformer | 896 | 424 | 472 | 47.32% | 48.33% | 46.88% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 896 | 399 | 497 | 44.53% | 44.58% | 42.50% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 896 | 399 | 497 | 44.53% | 45.42% | 44.17% | 5.47 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 896 | 383 | 513 | 42.75% | 39.17% | 42.29% | 7.25 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 896 | 377 | 519 | 42.08% | 42.92% | 41.88% | 7.92 pp | -142 | 47 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 718 | 347 | 371 | 48.33% | 46.67% | 48.33% | 1.67 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 718 | 342 | 376 | 47.63% | 45.42% | 50.00% | 2.37 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 718 | 332 | 386 | 46.24% | 43.33% | 47.71% | 3.76 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 718 | 310 | 408 | 43.18% | 37.50% | 42.08% | 6.82 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 718 | 305 | 413 | 42.48% | 40.00% | 43.33% | 7.52 pp | -108 | 43 | -2.51 |
| BTC Daily | xgb | XGBoost | 728 | 288 | 440 | 39.56% | 35.42% | 38.54% | 10.44 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 490 | 236 | 254 | 48.16% | 44.17% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 490 | 230 | 260 | 46.94% | 49.17% | 47.50% | 3.06 pp | -30 | 47 | -0.64 |
| BTC Market Hours | transformer | Transformer | 490 | 229 | 261 | 46.73% | 42.92% | 47.29% | 3.27 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 490 | 211 | 279 | 43.06% | 40.83% | 43.12% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 490 | 211 | 279 | 43.06% | 42.92% | 43.33% | 6.94 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 490 | 200 | 290 | 40.82% | 40.00% | 40.83% | 9.18 pp | -90 | 47 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 544 | 252 | 292 | 46.32% | 49.17% | 47.50% | 3.68 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 544 | 249 | 295 | 45.77% | 47.50% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 544 | 249 | 295 | 45.77% | 44.17% | 46.88% | 4.23 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 544 | 225 | 319 | 41.36% | 41.67% | 41.25% | 8.64 pp | -94 | 47 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 544 | 219 | 325 | 40.26% | 37.92% | 40.83% | 9.74 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 544 | 217 | 327 | 39.89% | 40.00% | 39.38% | 10.11 pp | -110 | 47 | -2.34 |

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
