# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T18:34:48.639043+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1131 | 843 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1007 | 642 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 603 | 404 | 198 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 605 | 458 | 145 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 1 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 1 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 404 | 201 | 203 | 49.75% | 49.17% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| BTC Daily | transformer | Transformer | 632 | 310 | 322 | 49.05% | 47.92% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 632 | 308 | 324 | 48.73% | 46.25% | 50.00% | 1.27 pp | -16 | 39 | -0.41 |
| BTC Market Hours | nn | NN | 404 | 192 | 212 | 47.52% | 51.25% | 47.52% | 2.48 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 404 | 187 | 217 | 46.29% | 42.92% | 46.29% | 3.71 pp | -30 | 40 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 458 | 212 | 246 | 46.29% | 46.25% | 46.29% | 3.71 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 458 | 210 | 248 | 45.85% | 47.08% | 45.85% | 4.15 pp | -38 | 40 | -0.95 |
| BTC Daily | nn | NN | 632 | 297 | 335 | 46.99% | 43.33% | 48.96% | 3.01 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 458 | 209 | 249 | 45.63% | 45.83% | 45.63% | 4.37 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 809 | 381 | 428 | 47.10% | 44.17% | 46.88% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 809 | 381 | 428 | 47.10% | 44.58% | 46.04% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 404 | 179 | 225 | 44.31% | 46.25% | 44.31% | 5.69 pp | -46 | 40 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 404 | 174 | 230 | 43.07% | 42.50% | 43.07% | 6.93 pp | -56 | 40 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| BTC Daily | lstm | LSTM | 632 | 280 | 352 | 44.30% | 42.50% | 43.75% | 5.70 pp | -72 | 39 | -1.85 |
| BTC Hourly | nn | NN | 809 | 363 | 446 | 44.87% | 40.42% | 44.58% | 5.13 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 404 | 163 | 241 | 40.35% | 38.75% | 40.35% | 9.65 pp | -78 | 40 | -1.95 |
| BTC Hourly | rf | RandomForest | 809 | 361 | 448 | 44.62% | 44.17% | 44.38% | 5.38 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 458 | 188 | 270 | 41.05% | 40.83% | 41.05% | 8.95 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 458 | 187 | 271 | 40.83% | 40.00% | 40.83% | 9.17 pp | -84 | 40 | -2.10 |
| BTC Hourly | lstm | LSTM | 809 | 354 | 455 | 43.76% | 42.08% | 44.58% | 6.24 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 632 | 270 | 362 | 42.72% | 42.50% | 43.54% | 7.28 pp | -92 | 39 | -2.36 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 458 | 180 | 278 | 39.30% | 36.67% | 39.30% | 10.70 pp | -98 | 40 | -2.45 |
| BTC Hourly | xgb | XGBoost | 809 | 343 | 466 | 42.40% | 40.00% | 42.92% | 7.60 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 642 | 251 | 391 | 39.10% | 30.83% | 39.17% | 10.90 pp | -140 | 39 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 809 | 381 | 428 | 47.10% | 44.17% | 46.88% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 809 | 381 | 428 | 47.10% | 44.58% | 46.04% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 809 | 363 | 446 | 44.87% | 40.42% | 44.58% | 5.13 pp | -83 | 44 | -1.89 |
| BTC Hourly | rf | RandomForest | 809 | 361 | 448 | 44.62% | 44.17% | 44.38% | 5.38 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 809 | 354 | 455 | 43.76% | 42.08% | 44.58% | 6.24 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 809 | 343 | 466 | 42.40% | 40.00% | 42.92% | 7.60 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 632 | 310 | 322 | 49.05% | 47.92% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 632 | 308 | 324 | 48.73% | 46.25% | 50.00% | 1.27 pp | -16 | 39 | -0.41 |
| BTC Daily | nn | NN | 632 | 297 | 335 | 46.99% | 43.33% | 48.96% | 3.01 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 632 | 280 | 352 | 44.30% | 42.50% | 43.75% | 5.70 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 632 | 270 | 362 | 42.72% | 42.50% | 43.54% | 7.28 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 642 | 251 | 391 | 39.10% | 30.83% | 39.17% | 10.90 pp | -140 | 39 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 404 | 201 | 203 | 49.75% | 49.17% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Market Hours | nn | NN | 404 | 192 | 212 | 47.52% | 51.25% | 47.52% | 2.48 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 404 | 187 | 217 | 46.29% | 42.92% | 46.29% | 3.71 pp | -30 | 40 | -0.75 |
| BTC Market Hours | lstm | LSTM | 404 | 179 | 225 | 44.31% | 46.25% | 44.31% | 5.69 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 404 | 174 | 230 | 43.07% | 42.50% | 43.07% | 6.93 pp | -56 | 40 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 404 | 163 | 241 | 40.35% | 38.75% | 40.35% | 9.65 pp | -78 | 40 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 458 | 212 | 246 | 46.29% | 46.25% | 46.29% | 3.71 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 458 | 210 | 248 | 45.85% | 47.08% | 45.85% | 4.15 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 458 | 209 | 249 | 45.63% | 45.83% | 45.63% | 4.37 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 458 | 188 | 270 | 41.05% | 40.83% | 41.05% | 8.95 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 458 | 187 | 271 | 40.83% | 40.00% | 40.83% | 9.17 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 458 | 180 | 278 | 39.30% | 36.67% | 39.30% | 10.70 pp | -98 | 40 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
