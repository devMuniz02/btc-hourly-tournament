# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T03:11:19.855403+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1318 | 1030 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1193 | 828 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 939 | 590 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 941 | 644 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 590 | 286 | 304 | 48.47% | 47.92% | 47.29% | 1.53 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 590 | 285 | 305 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| BTC Market Hours | transformer | Transformer | 590 | 277 | 313 | 46.95% | 46.67% | 46.25% | 3.05 pp | -36 | 55 | -0.65 |
| BTC Market Hours Daily | nn | NN | 644 | 303 | 341 | 47.05% | 49.17% | 48.33% | 2.95 pp | -38 | 55 | -0.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 818 | 392 | 426 | 47.92% | 45.42% | 46.88% | 2.08 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 644 | 302 | 342 | 46.89% | 49.17% | 47.29% | 3.11 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 644 | 301 | 343 | 46.74% | 49.17% | 47.71% | 3.26 pp | -42 | 55 | -0.76 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 996 | 472 | 524 | 47.39% | 48.75% | 46.25% | 2.61 pp | -52 | 51 | -1.02 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 818 | 381 | 437 | 46.58% | 45.42% | 45.42% | 3.42 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 818 | 379 | 439 | 46.33% | 39.58% | 45.83% | 3.67 pp | -60 | 47 | -1.28 |
| BTC Hourly | transformer | Transformer | 996 | 463 | 533 | 46.49% | 45.00% | 44.58% | 3.51 pp | -70 | 51 | -1.37 |
| BTC Market Hours | xgb | XGBoost | 590 | 252 | 338 | 42.71% | 44.17% | 43.33% | 7.29 pp | -86 | 55 | -1.56 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 590 | 251 | 339 | 42.54% | 42.08% | 42.50% | 7.46 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 590 | 251 | 339 | 42.54% | 42.92% | 42.50% | 7.46 pp | -88 | 55 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 644 | 266 | 378 | 41.30% | 42.08% | 40.62% | 8.70 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 644 | 265 | 379 | 41.15% | 43.33% | 40.21% | 8.85 pp | -114 | 55 | -2.07 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 644 | 263 | 381 | 40.84% | 41.67% | 40.42% | 9.16 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 996 | 440 | 556 | 44.18% | 42.50% | 42.08% | 5.82 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 996 | 439 | 557 | 44.08% | 41.25% | 43.12% | 5.92 pp | -118 | 51 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| BTC Daily | lstm | LSTM | 818 | 347 | 471 | 42.42% | 36.67% | 41.04% | 7.58 pp | -124 | 47 | -2.64 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 818 | 339 | 479 | 41.44% | 37.92% | 40.83% | 8.56 pp | -140 | 47 | -2.98 |
| BTC Hourly | lstm | LSTM | 996 | 422 | 574 | 42.37% | 37.08% | 40.00% | 7.63 pp | -152 | 51 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 996 | 410 | 586 | 41.16% | 35.00% | 38.54% | 8.84 pp | -176 | 51 | -3.45 |
| BTC Daily | xgb | XGBoost | 828 | 325 | 503 | 39.25% | 37.50% | 36.04% | 10.75 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 996 | 472 | 524 | 47.39% | 48.75% | 46.25% | 2.61 pp | -52 | 51 | -1.02 |
| BTC Hourly | transformer | Transformer | 996 | 463 | 533 | 46.49% | 45.00% | 44.58% | 3.51 pp | -70 | 51 | -1.37 |
| BTC Hourly | nn | NN | 996 | 440 | 556 | 44.18% | 42.50% | 42.08% | 5.82 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 996 | 439 | 557 | 44.08% | 41.25% | 43.12% | 5.92 pp | -118 | 51 | -2.31 |
| BTC Hourly | lstm | LSTM | 996 | 422 | 574 | 42.37% | 37.08% | 40.00% | 7.63 pp | -152 | 51 | -2.98 |
| BTC Hourly | xgb | XGBoost | 996 | 410 | 586 | 41.16% | 35.00% | 38.54% | 8.84 pp | -176 | 51 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 818 | 392 | 426 | 47.92% | 45.42% | 46.88% | 2.08 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 818 | 381 | 437 | 46.58% | 45.42% | 45.42% | 3.42 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 818 | 379 | 439 | 46.33% | 39.58% | 45.83% | 3.67 pp | -60 | 47 | -1.28 |
| BTC Daily | lstm | LSTM | 818 | 347 | 471 | 42.42% | 36.67% | 41.04% | 7.58 pp | -124 | 47 | -2.64 |
| BTC Daily | rf | RandomForest | 818 | 339 | 479 | 41.44% | 37.92% | 40.83% | 8.56 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 828 | 325 | 503 | 39.25% | 37.50% | 36.04% | 10.75 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 590 | 286 | 304 | 48.47% | 47.92% | 47.29% | 1.53 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 590 | 285 | 305 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 590 | 277 | 313 | 46.95% | 46.67% | 46.25% | 3.05 pp | -36 | 55 | -0.65 |
| BTC Market Hours | xgb | XGBoost | 590 | 252 | 338 | 42.71% | 44.17% | 43.33% | 7.29 pp | -86 | 55 | -1.56 |
| BTC Market Hours | lstm | LSTM | 590 | 251 | 339 | 42.54% | 42.08% | 42.50% | 7.46 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 590 | 251 | 339 | 42.54% | 42.92% | 42.50% | 7.46 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 644 | 303 | 341 | 47.05% | 49.17% | 48.33% | 2.95 pp | -38 | 55 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 644 | 302 | 342 | 46.89% | 49.17% | 47.29% | 3.11 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 644 | 301 | 343 | 46.74% | 49.17% | 47.71% | 3.26 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 644 | 266 | 378 | 41.30% | 42.08% | 40.62% | 8.70 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 644 | 265 | 379 | 41.15% | 43.33% | 40.21% | 8.85 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 644 | 263 | 381 | 40.84% | 41.67% | 40.42% | 9.16 pp | -118 | 55 | -2.15 |

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
