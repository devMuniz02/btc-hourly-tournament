# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T18:52:27.616684+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1281 | 993 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1156 | 791 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 17:00:00+00:00 | 869 | 553 | 315 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 17:00:00+00:00 | 871 | 607 | 262 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 553 | 269 | 284 | 48.64% | 47.08% | 47.92% | 1.36 pp | -15 | 52 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 553 | 264 | 289 | 47.74% | 51.25% | 49.79% | 2.26 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 553 | 262 | 291 | 47.38% | 47.92% | 47.92% | 2.62 pp | -29 | 52 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 781 | 375 | 406 | 48.02% | 45.42% | 47.29% | 1.98 pp | -31 | 45 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 607 | 285 | 322 | 46.95% | 49.58% | 47.71% | 3.05 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 607 | 283 | 324 | 46.62% | 47.92% | 48.12% | 3.38 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 607 | 281 | 326 | 46.29% | 48.75% | 46.88% | 3.71 pp | -45 | 52 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 959 | 457 | 502 | 47.65% | 49.17% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 781 | 362 | 419 | 46.35% | 40.42% | 46.46% | 3.65 pp | -57 | 45 | -1.27 |
| BTC Hourly | transformer | Transformer | 959 | 447 | 512 | 46.61% | 45.00% | 43.96% | 3.39 pp | -65 | 50 | -1.30 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| BTC Daily | nn | NN | 781 | 361 | 420 | 46.22% | 44.17% | 45.00% | 3.78 pp | -59 | 45 | -1.31 |
| BTC Market Hours | rf | RandomForest | 553 | 239 | 314 | 43.22% | 45.42% | 43.12% | 6.78 pp | -75 | 52 | -1.44 |
| BTC Market Hours | lstm | LSTM | 553 | 237 | 316 | 42.86% | 41.25% | 43.12% | 7.14 pp | -79 | 52 | -1.52 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 553 | 232 | 321 | 41.95% | 44.58% | 42.08% | 8.05 pp | -89 | 52 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 607 | 254 | 353 | 41.85% | 43.75% | 41.25% | 8.15 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 959 | 425 | 534 | 44.32% | 43.33% | 43.33% | 5.68 pp | -109 | 50 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 607 | 246 | 361 | 40.53% | 40.00% | 40.83% | 9.47 pp | -115 | 52 | -2.21 |
| BTC Hourly | nn | NN | 959 | 424 | 535 | 44.21% | 42.08% | 42.50% | 5.79 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 607 | 245 | 362 | 40.36% | 41.25% | 39.79% | 9.64 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 781 | 328 | 453 | 42.00% | 34.17% | 39.58% | 8.00 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 959 | 409 | 550 | 42.65% | 37.50% | 42.08% | 7.35 pp | -141 | 50 | -2.82 |
| BTC Daily | rf | RandomForest | 781 | 326 | 455 | 41.74% | 37.92% | 41.67% | 8.26 pp | -129 | 45 | -2.87 |
| BTC Hourly | xgb | XGBoost | 959 | 397 | 562 | 41.40% | 37.92% | 39.58% | 8.60 pp | -165 | 50 | -3.30 |
| BTC Daily | xgb | XGBoost | 791 | 307 | 484 | 38.81% | 34.58% | 36.04% | 11.19 pp | -177 | 45 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 959 | 457 | 502 | 47.65% | 49.17% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 959 | 447 | 512 | 46.61% | 45.00% | 43.96% | 3.39 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 959 | 425 | 534 | 44.32% | 43.33% | 43.33% | 5.68 pp | -109 | 50 | -2.18 |
| BTC Hourly | nn | NN | 959 | 424 | 535 | 44.21% | 42.08% | 42.50% | 5.79 pp | -111 | 50 | -2.22 |
| BTC Hourly | lstm | LSTM | 959 | 409 | 550 | 42.65% | 37.50% | 42.08% | 7.35 pp | -141 | 50 | -2.82 |
| BTC Hourly | xgb | XGBoost | 959 | 397 | 562 | 41.40% | 37.92% | 39.58% | 8.60 pp | -165 | 50 | -3.30 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 781 | 375 | 406 | 48.02% | 45.42% | 47.29% | 1.98 pp | -31 | 45 | -0.69 |
| BTC Daily | transformer | Transformer | 781 | 362 | 419 | 46.35% | 40.42% | 46.46% | 3.65 pp | -57 | 45 | -1.27 |
| BTC Daily | nn | NN | 781 | 361 | 420 | 46.22% | 44.17% | 45.00% | 3.78 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 781 | 328 | 453 | 42.00% | 34.17% | 39.58% | 8.00 pp | -125 | 45 | -2.78 |
| BTC Daily | rf | RandomForest | 781 | 326 | 455 | 41.74% | 37.92% | 41.67% | 8.26 pp | -129 | 45 | -2.87 |
| BTC Daily | xgb | XGBoost | 791 | 307 | 484 | 38.81% | 34.58% | 36.04% | 11.19 pp | -177 | 45 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 553 | 269 | 284 | 48.64% | 47.08% | 47.92% | 1.36 pp | -15 | 52 | -0.29 |
| BTC Market Hours | nn | NN | 553 | 264 | 289 | 47.74% | 51.25% | 49.79% | 2.26 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 553 | 262 | 291 | 47.38% | 47.92% | 47.92% | 2.62 pp | -29 | 52 | -0.56 |
| BTC Market Hours | rf | RandomForest | 553 | 239 | 314 | 43.22% | 45.42% | 43.12% | 6.78 pp | -75 | 52 | -1.44 |
| BTC Market Hours | lstm | LSTM | 553 | 237 | 316 | 42.86% | 41.25% | 43.12% | 7.14 pp | -79 | 52 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 553 | 232 | 321 | 41.95% | 44.58% | 42.08% | 8.05 pp | -89 | 52 | -1.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 607 | 285 | 322 | 46.95% | 49.58% | 47.71% | 3.05 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 607 | 283 | 324 | 46.62% | 47.92% | 48.12% | 3.38 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 607 | 281 | 326 | 46.29% | 48.75% | 46.88% | 3.71 pp | -45 | 52 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 607 | 254 | 353 | 41.85% | 43.75% | 41.25% | 8.15 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 607 | 246 | 361 | 40.53% | 40.00% | 40.83% | 9.47 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 607 | 245 | 362 | 40.36% | 41.25% | 39.79% | 9.64 pp | -117 | 52 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
