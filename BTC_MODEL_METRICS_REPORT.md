# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T19:47:15.347979+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1313 | 1025 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1189 | 824 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 18:00:00+00:00 | 929 | 586 | 342 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 18:00:00+00:00 | 931 | 640 | 289 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 228 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 586 | 285 | 301 | 48.63% | 47.92% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 586 | 283 | 303 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| BTC Market Hours | transformer | Transformer | 586 | 276 | 310 | 47.10% | 47.08% | 46.67% | 2.90 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 814 | 392 | 422 | 48.16% | 45.83% | 47.08% | 1.84 pp | -30 | 47 | -0.64 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 640 | 301 | 339 | 47.03% | 49.58% | 47.50% | 2.97 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 640 | 300 | 340 | 46.88% | 48.33% | 48.12% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 640 | 300 | 340 | 46.88% | 48.75% | 47.50% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 991 | 471 | 520 | 47.53% | 49.58% | 46.46% | 2.47 pp | -49 | 51 | -0.96 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 814 | 379 | 435 | 46.56% | 45.00% | 45.42% | 3.44 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 814 | 378 | 436 | 46.44% | 40.00% | 46.25% | 3.56 pp | -58 | 47 | -1.23 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 991 | 460 | 531 | 46.42% | 44.58% | 43.96% | 3.58 pp | -71 | 51 | -1.39 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 586 | 251 | 335 | 42.83% | 43.33% | 43.12% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 586 | 251 | 335 | 42.83% | 44.17% | 43.33% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | lstm | LSTM | 586 | 250 | 336 | 42.66% | 42.08% | 42.71% | 7.34 pp | -86 | 54 | -1.59 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 640 | 266 | 374 | 41.56% | 42.50% | 41.04% | 8.44 pp | -108 | 54 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 640 | 264 | 376 | 41.25% | 43.33% | 40.62% | 8.75 pp | -112 | 54 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 640 | 263 | 377 | 41.09% | 42.08% | 41.04% | 8.91 pp | -114 | 54 | -2.11 |
| BTC Hourly | nn | NN | 991 | 437 | 554 | 44.10% | 42.08% | 42.08% | 5.90 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 991 | 437 | 554 | 44.10% | 41.25% | 42.71% | 5.90 pp | -117 | 51 | -2.29 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 814 | 344 | 470 | 42.26% | 35.83% | 40.62% | 7.74 pp | -126 | 47 | -2.68 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 991 | 420 | 571 | 42.38% | 36.67% | 40.00% | 7.62 pp | -151 | 51 | -2.96 |
| BTC Daily | rf | RandomForest | 814 | 337 | 477 | 41.40% | 37.08% | 40.83% | 8.60 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 991 | 407 | 584 | 41.07% | 35.00% | 38.33% | 8.93 pp | -177 | 51 | -3.47 |
| BTC Daily | xgb | XGBoost | 824 | 323 | 501 | 39.20% | 36.25% | 36.04% | 10.80 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 991 | 471 | 520 | 47.53% | 49.58% | 46.46% | 2.47 pp | -49 | 51 | -0.96 |
| BTC Hourly | transformer | Transformer | 991 | 460 | 531 | 46.42% | 44.58% | 43.96% | 3.58 pp | -71 | 51 | -1.39 |
| BTC Hourly | nn | NN | 991 | 437 | 554 | 44.10% | 42.08% | 42.08% | 5.90 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 991 | 437 | 554 | 44.10% | 41.25% | 42.71% | 5.90 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 991 | 420 | 571 | 42.38% | 36.67% | 40.00% | 7.62 pp | -151 | 51 | -2.96 |
| BTC Hourly | xgb | XGBoost | 991 | 407 | 584 | 41.07% | 35.00% | 38.33% | 8.93 pp | -177 | 51 | -3.47 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 814 | 392 | 422 | 48.16% | 45.83% | 47.08% | 1.84 pp | -30 | 47 | -0.64 |
| BTC Daily | nn | NN | 814 | 379 | 435 | 46.56% | 45.00% | 45.42% | 3.44 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 814 | 378 | 436 | 46.44% | 40.00% | 46.25% | 3.56 pp | -58 | 47 | -1.23 |
| BTC Daily | lstm | LSTM | 814 | 344 | 470 | 42.26% | 35.83% | 40.62% | 7.74 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 814 | 337 | 477 | 41.40% | 37.08% | 40.83% | 8.60 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 824 | 323 | 501 | 39.20% | 36.25% | 36.04% | 10.80 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 586 | 285 | 301 | 48.63% | 47.92% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 586 | 283 | 303 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 586 | 276 | 310 | 47.10% | 47.08% | 46.67% | 2.90 pp | -34 | 54 | -0.63 |
| BTC Market Hours | rf | RandomForest | 586 | 251 | 335 | 42.83% | 43.33% | 43.12% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 586 | 251 | 335 | 42.83% | 44.17% | 43.33% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | lstm | LSTM | 586 | 250 | 336 | 42.66% | 42.08% | 42.71% | 7.34 pp | -86 | 54 | -1.59 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 640 | 301 | 339 | 47.03% | 49.58% | 47.50% | 2.97 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 640 | 300 | 340 | 46.88% | 48.33% | 48.12% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 640 | 300 | 340 | 46.88% | 48.75% | 47.50% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | rf | RandomForest | 640 | 266 | 374 | 41.56% | 42.50% | 41.04% | 8.44 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 640 | 264 | 376 | 41.25% | 43.33% | 40.62% | 8.75 pp | -112 | 54 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 640 | 263 | 377 | 41.09% | 42.08% | 41.04% | 8.91 pp | -114 | 54 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
