# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T12:22:02.246204+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1276 | 988 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1152 | 787 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 859 | 549 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 861 | 603 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 193 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 549 | 266 | 283 | 48.45% | 45.83% | 47.50% | 1.55 pp | -17 | 52 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 549 | 262 | 287 | 47.72% | 51.25% | 49.58% | 2.28 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 549 | 260 | 289 | 47.36% | 47.92% | 47.92% | 2.64 pp | -29 | 52 | -0.56 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 777 | 374 | 403 | 48.13% | 45.83% | 47.29% | 1.87 pp | -29 | 45 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 603 | 284 | 319 | 47.10% | 50.00% | 48.12% | 2.90 pp | -35 | 52 | -0.67 |
| BTC Market Hours Daily | nn | NN | 603 | 279 | 324 | 46.27% | 46.25% | 47.50% | 3.73 pp | -45 | 52 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 603 | 278 | 325 | 46.10% | 48.33% | 46.88% | 3.90 pp | -47 | 52 | -0.90 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 954 | 454 | 500 | 47.59% | 49.17% | 47.08% | 2.41 pp | -46 | 50 | -0.92 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| BTC Daily | transformer | Transformer | 777 | 362 | 415 | 46.59% | 41.67% | 46.88% | 3.41 pp | -53 | 45 | -1.18 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 777 | 359 | 418 | 46.20% | 44.17% | 45.21% | 3.80 pp | -59 | 45 | -1.31 |
| BTC Hourly | transformer | Transformer | 954 | 444 | 510 | 46.54% | 44.17% | 44.17% | 3.46 pp | -66 | 50 | -1.32 |
| BTC Market Hours | rf | RandomForest | 549 | 238 | 311 | 43.35% | 45.42% | 43.33% | 6.65 pp | -73 | 52 | -1.40 |
| Consolidated Hourly | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| BTC Market Hours | lstm | LSTM | 549 | 235 | 314 | 42.81% | 41.25% | 43.12% | 7.19 pp | -79 | 52 | -1.52 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 549 | 230 | 319 | 41.89% | 43.75% | 41.88% | 8.11 pp | -89 | 52 | -1.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 603 | 253 | 350 | 41.96% | 44.17% | 41.46% | 8.04 pp | -97 | 52 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 954 | 423 | 531 | 44.34% | 42.92% | 43.33% | 5.66 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 954 | 422 | 532 | 44.23% | 42.50% | 42.71% | 5.77 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 603 | 242 | 361 | 40.13% | 38.75% | 40.00% | 9.87 pp | -119 | 52 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 603 | 242 | 361 | 40.13% | 40.83% | 39.38% | 9.87 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 777 | 327 | 450 | 42.08% | 35.00% | 39.58% | 7.92 pp | -123 | 45 | -2.73 |
| BTC Hourly | lstm | LSTM | 954 | 408 | 546 | 42.77% | 37.08% | 42.29% | 7.23 pp | -138 | 50 | -2.76 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Daily | rf | RandomForest | 777 | 325 | 452 | 41.83% | 38.75% | 41.88% | 8.17 pp | -127 | 45 | -2.82 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 954 | 397 | 557 | 41.61% | 38.75% | 40.21% | 8.39 pp | -160 | 50 | -3.20 |
| BTC Daily | xgb | XGBoost | 787 | 307 | 480 | 39.01% | 35.42% | 36.25% | 10.99 pp | -173 | 45 | -3.84 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 954 | 454 | 500 | 47.59% | 49.17% | 47.08% | 2.41 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 954 | 444 | 510 | 46.54% | 44.17% | 44.17% | 3.46 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 954 | 423 | 531 | 44.34% | 42.92% | 43.33% | 5.66 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 954 | 422 | 532 | 44.23% | 42.50% | 42.71% | 5.77 pp | -110 | 50 | -2.20 |
| BTC Hourly | lstm | LSTM | 954 | 408 | 546 | 42.77% | 37.08% | 42.29% | 7.23 pp | -138 | 50 | -2.76 |
| BTC Hourly | xgb | XGBoost | 954 | 397 | 557 | 41.61% | 38.75% | 40.21% | 8.39 pp | -160 | 50 | -3.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 777 | 374 | 403 | 48.13% | 45.83% | 47.29% | 1.87 pp | -29 | 45 | -0.64 |
| BTC Daily | transformer | Transformer | 777 | 362 | 415 | 46.59% | 41.67% | 46.88% | 3.41 pp | -53 | 45 | -1.18 |
| BTC Daily | nn | NN | 777 | 359 | 418 | 46.20% | 44.17% | 45.21% | 3.80 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 777 | 327 | 450 | 42.08% | 35.00% | 39.58% | 7.92 pp | -123 | 45 | -2.73 |
| BTC Daily | rf | RandomForest | 777 | 325 | 452 | 41.83% | 38.75% | 41.88% | 8.17 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 787 | 307 | 480 | 39.01% | 35.42% | 36.25% | 10.99 pp | -173 | 45 | -3.84 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 549 | 266 | 283 | 48.45% | 45.83% | 47.50% | 1.55 pp | -17 | 52 | -0.33 |
| BTC Market Hours | nn | NN | 549 | 262 | 287 | 47.72% | 51.25% | 49.58% | 2.28 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 549 | 260 | 289 | 47.36% | 47.92% | 47.92% | 2.64 pp | -29 | 52 | -0.56 |
| BTC Market Hours | rf | RandomForest | 549 | 238 | 311 | 43.35% | 45.42% | 43.33% | 6.65 pp | -73 | 52 | -1.40 |
| BTC Market Hours | lstm | LSTM | 549 | 235 | 314 | 42.81% | 41.25% | 43.12% | 7.19 pp | -79 | 52 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 549 | 230 | 319 | 41.89% | 43.75% | 41.88% | 8.11 pp | -89 | 52 | -1.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 603 | 284 | 319 | 47.10% | 50.00% | 48.12% | 2.90 pp | -35 | 52 | -0.67 |
| BTC Market Hours Daily | nn | NN | 603 | 279 | 324 | 46.27% | 46.25% | 47.50% | 3.73 pp | -45 | 52 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 603 | 278 | 325 | 46.10% | 48.33% | 46.88% | 3.90 pp | -47 | 52 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 603 | 253 | 350 | 41.96% | 44.17% | 41.46% | 8.04 pp | -97 | 52 | -1.87 |
| BTC Market Hours Daily | lstm | LSTM | 603 | 242 | 361 | 40.13% | 38.75% | 40.00% | 9.87 pp | -119 | 52 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 603 | 242 | 361 | 40.13% | 40.83% | 39.38% | 9.87 pp | -119 | 52 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 94 | 99 | 48.70% | 48.70% | 48.70% | 1.30 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
