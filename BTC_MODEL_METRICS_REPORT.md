# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T04:43:18.688888+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1319 | 1031 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1194 | 829 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 940 | 591 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 942 | 645 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 591 | 286 | 305 | 48.39% | 47.50% | 47.08% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 591 | 285 | 306 | 48.22% | 52.08% | 49.79% | 1.78 pp | -21 | 55 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| BTC Market Hours | transformer | Transformer | 591 | 277 | 314 | 46.87% | 46.67% | 46.25% | 3.13 pp | -37 | 55 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 645 | 303 | 342 | 46.98% | 49.17% | 47.50% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | nn | NN | 645 | 303 | 342 | 46.98% | 48.75% | 48.12% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 819 | 392 | 427 | 47.86% | 45.00% | 46.88% | 2.14 pp | -35 | 47 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 645 | 301 | 344 | 46.67% | 48.75% | 47.71% | 3.33 pp | -43 | 55 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 997 | 473 | 524 | 47.44% | 49.17% | 46.25% | 2.56 pp | -51 | 52 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 819 | 381 | 438 | 46.52% | 45.00% | 45.42% | 3.48 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 819 | 379 | 440 | 46.28% | 39.17% | 45.62% | 3.72 pp | -61 | 47 | -1.30 |
| BTC Hourly | transformer | Transformer | 997 | 463 | 534 | 46.44% | 45.00% | 44.58% | 3.56 pp | -71 | 52 | -1.37 |
| BTC Market Hours | xgb | XGBoost | 591 | 253 | 338 | 42.81% | 44.58% | 43.33% | 7.19 pp | -85 | 55 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 591 | 252 | 339 | 42.64% | 42.92% | 42.50% | 7.36 pp | -87 | 55 | -1.58 |
| BTC Market Hours | lstm | LSTM | 591 | 251 | 340 | 42.47% | 42.08% | 42.29% | 7.53 pp | -89 | 55 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 645 | 266 | 379 | 41.24% | 41.67% | 40.62% | 8.76 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 645 | 265 | 380 | 41.09% | 42.92% | 40.21% | 8.91 pp | -115 | 55 | -2.09 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 645 | 263 | 382 | 40.78% | 41.67% | 40.21% | 9.22 pp | -119 | 55 | -2.16 |
| BTC Hourly | nn | NN | 997 | 440 | 557 | 44.13% | 42.50% | 41.88% | 5.87 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 997 | 439 | 558 | 44.03% | 41.25% | 43.12% | 5.97 pp | -119 | 52 | -2.29 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| BTC Daily | lstm | LSTM | 819 | 348 | 471 | 42.49% | 36.67% | 41.25% | 7.51 pp | -123 | 47 | -2.62 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| BTC Hourly | lstm | LSTM | 997 | 423 | 574 | 42.43% | 37.50% | 40.21% | 7.57 pp | -151 | 52 | -2.90 |
| BTC Daily | rf | RandomForest | 819 | 340 | 479 | 41.51% | 37.92% | 41.04% | 8.49 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 997 | 410 | 587 | 41.12% | 35.00% | 38.54% | 8.88 pp | -177 | 52 | -3.40 |
| BTC Daily | xgb | XGBoost | 829 | 326 | 503 | 39.32% | 37.50% | 36.25% | 10.68 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 997 | 473 | 524 | 47.44% | 49.17% | 46.25% | 2.56 pp | -51 | 52 | -0.98 |
| BTC Hourly | transformer | Transformer | 997 | 463 | 534 | 46.44% | 45.00% | 44.58% | 3.56 pp | -71 | 52 | -1.37 |
| BTC Hourly | nn | NN | 997 | 440 | 557 | 44.13% | 42.50% | 41.88% | 5.87 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 997 | 439 | 558 | 44.03% | 41.25% | 43.12% | 5.97 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 997 | 423 | 574 | 42.43% | 37.50% | 40.21% | 7.57 pp | -151 | 52 | -2.90 |
| BTC Hourly | xgb | XGBoost | 997 | 410 | 587 | 41.12% | 35.00% | 38.54% | 8.88 pp | -177 | 52 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 819 | 392 | 427 | 47.86% | 45.00% | 46.88% | 2.14 pp | -35 | 47 | -0.74 |
| BTC Daily | nn | NN | 819 | 381 | 438 | 46.52% | 45.00% | 45.42% | 3.48 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 819 | 379 | 440 | 46.28% | 39.17% | 45.62% | 3.72 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 819 | 348 | 471 | 42.49% | 36.67% | 41.25% | 7.51 pp | -123 | 47 | -2.62 |
| BTC Daily | rf | RandomForest | 819 | 340 | 479 | 41.51% | 37.92% | 41.04% | 8.49 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 829 | 326 | 503 | 39.32% | 37.50% | 36.25% | 10.68 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 591 | 286 | 305 | 48.39% | 47.50% | 47.08% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 591 | 285 | 306 | 48.22% | 52.08% | 49.79% | 1.78 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 591 | 277 | 314 | 46.87% | 46.67% | 46.25% | 3.13 pp | -37 | 55 | -0.67 |
| BTC Market Hours | xgb | XGBoost | 591 | 253 | 338 | 42.81% | 44.58% | 43.33% | 7.19 pp | -85 | 55 | -1.55 |
| BTC Market Hours | rf | RandomForest | 591 | 252 | 339 | 42.64% | 42.92% | 42.50% | 7.36 pp | -87 | 55 | -1.58 |
| BTC Market Hours | lstm | LSTM | 591 | 251 | 340 | 42.47% | 42.08% | 42.29% | 7.53 pp | -89 | 55 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 645 | 303 | 342 | 46.98% | 49.17% | 47.50% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | nn | NN | 645 | 303 | 342 | 46.98% | 48.75% | 48.12% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 645 | 301 | 344 | 46.67% | 48.75% | 47.71% | 3.33 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 645 | 266 | 379 | 41.24% | 41.67% | 40.62% | 8.76 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 645 | 265 | 380 | 41.09% | 42.92% | 40.21% | 8.91 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 645 | 263 | 382 | 40.78% | 41.67% | 40.21% | 9.22 pp | -119 | 55 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
