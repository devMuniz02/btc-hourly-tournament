# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T13:43:52.347444+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1277 | 989 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1153 | 788 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 861 | 550 | 310 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 863 | 604 | 257 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 195 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 550 | 266 | 284 | 48.36% | 45.83% | 47.29% | 1.64 pp | -18 | 52 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 550 | 262 | 288 | 47.64% | 51.25% | 49.58% | 2.36 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 550 | 260 | 290 | 47.27% | 47.50% | 47.92% | 2.73 pp | -30 | 52 | -0.58 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 778 | 374 | 404 | 48.07% | 45.42% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 604 | 284 | 320 | 47.02% | 50.00% | 47.92% | 2.98 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 604 | 280 | 324 | 46.36% | 46.67% | 47.71% | 3.64 pp | -44 | 52 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 604 | 279 | 325 | 46.19% | 48.75% | 47.08% | 3.81 pp | -46 | 52 | -0.88 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 955 | 455 | 500 | 47.64% | 49.17% | 47.29% | 2.36 pp | -45 | 50 | -0.90 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| BTC Daily | transformer | Transformer | 778 | 362 | 416 | 46.53% | 41.67% | 46.67% | 3.47 pp | -54 | 45 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 778 | 359 | 419 | 46.14% | 44.17% | 45.00% | 3.86 pp | -60 | 45 | -1.33 |
| BTC Hourly | transformer | Transformer | 955 | 444 | 511 | 46.49% | 44.17% | 43.96% | 3.51 pp | -67 | 50 | -1.34 |
| Consolidated Hourly | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| BTC Market Hours | rf | RandomForest | 550 | 238 | 312 | 43.27% | 45.00% | 43.12% | 6.73 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 550 | 236 | 314 | 42.91% | 41.25% | 43.12% | 7.09 pp | -78 | 52 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 550 | 230 | 320 | 41.82% | 43.75% | 41.67% | 8.18 pp | -90 | 52 | -1.73 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 604 | 253 | 351 | 41.89% | 44.17% | 41.25% | 8.11 pp | -98 | 52 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |
| BTC Hourly | nn | NN | 955 | 423 | 532 | 44.29% | 42.50% | 42.92% | 5.71 pp | -109 | 50 | -2.18 |
| BTC Hourly | rf | RandomForest | 955 | 423 | 532 | 44.29% | 42.92% | 43.33% | 5.71 pp | -109 | 50 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 604 | 243 | 361 | 40.23% | 39.17% | 40.21% | 9.77 pp | -118 | 52 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 41.25% | 39.58% | 9.77 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Hourly | lstm | LSTM | 955 | 409 | 546 | 42.83% | 37.50% | 42.50% | 7.17 pp | -137 | 50 | -2.74 |
| BTC Daily | lstm | LSTM | 778 | 327 | 451 | 42.03% | 34.58% | 39.58% | 7.97 pp | -124 | 45 | -2.76 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Daily | rf | RandomForest | 778 | 325 | 453 | 41.77% | 38.75% | 41.67% | 8.23 pp | -128 | 45 | -2.84 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 955 | 397 | 558 | 41.57% | 38.75% | 40.00% | 8.43 pp | -161 | 50 | -3.22 |
| BTC Daily | xgb | XGBoost | 788 | 307 | 481 | 38.96% | 35.42% | 36.25% | 11.04 pp | -174 | 45 | -3.87 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 955 | 455 | 500 | 47.64% | 49.17% | 47.29% | 2.36 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 955 | 444 | 511 | 46.49% | 44.17% | 43.96% | 3.51 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 955 | 423 | 532 | 44.29% | 42.50% | 42.92% | 5.71 pp | -109 | 50 | -2.18 |
| BTC Hourly | rf | RandomForest | 955 | 423 | 532 | 44.29% | 42.92% | 43.33% | 5.71 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 955 | 409 | 546 | 42.83% | 37.50% | 42.50% | 7.17 pp | -137 | 50 | -2.74 |
| BTC Hourly | xgb | XGBoost | 955 | 397 | 558 | 41.57% | 38.75% | 40.00% | 8.43 pp | -161 | 50 | -3.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 778 | 374 | 404 | 48.07% | 45.42% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Daily | transformer | Transformer | 778 | 362 | 416 | 46.53% | 41.67% | 46.67% | 3.47 pp | -54 | 45 | -1.20 |
| BTC Daily | nn | NN | 778 | 359 | 419 | 46.14% | 44.17% | 45.00% | 3.86 pp | -60 | 45 | -1.33 |
| BTC Daily | lstm | LSTM | 778 | 327 | 451 | 42.03% | 34.58% | 39.58% | 7.97 pp | -124 | 45 | -2.76 |
| BTC Daily | rf | RandomForest | 778 | 325 | 453 | 41.77% | 38.75% | 41.67% | 8.23 pp | -128 | 45 | -2.84 |
| BTC Daily | xgb | XGBoost | 788 | 307 | 481 | 38.96% | 35.42% | 36.25% | 11.04 pp | -174 | 45 | -3.87 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 550 | 266 | 284 | 48.36% | 45.83% | 47.29% | 1.64 pp | -18 | 52 | -0.35 |
| BTC Market Hours | nn | NN | 550 | 262 | 288 | 47.64% | 51.25% | 49.58% | 2.36 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 550 | 260 | 290 | 47.27% | 47.50% | 47.92% | 2.73 pp | -30 | 52 | -0.58 |
| BTC Market Hours | rf | RandomForest | 550 | 238 | 312 | 43.27% | 45.00% | 43.12% | 6.73 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 550 | 236 | 314 | 42.91% | 41.25% | 43.12% | 7.09 pp | -78 | 52 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 550 | 230 | 320 | 41.82% | 43.75% | 41.67% | 8.18 pp | -90 | 52 | -1.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 604 | 284 | 320 | 47.02% | 50.00% | 47.92% | 2.98 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 604 | 280 | 324 | 46.36% | 46.67% | 47.71% | 3.64 pp | -44 | 52 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 604 | 279 | 325 | 46.19% | 48.75% | 47.08% | 3.81 pp | -46 | 52 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 604 | 253 | 351 | 41.89% | 44.17% | 41.25% | 8.11 pp | -98 | 52 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 604 | 243 | 361 | 40.23% | 39.17% | 40.21% | 9.77 pp | -118 | 52 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 41.25% | 39.58% | 9.77 pp | -118 | 52 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |

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
