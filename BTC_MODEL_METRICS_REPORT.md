# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T20:56:09.956277+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1314 | 1026 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1189 | 824 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 19:00:00+00:00 | 930 | 586 | 343 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 19:00:00+00:00 | 932 | 640 | 290 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 586 | 285 | 301 | 48.63% | 47.92% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| BTC Market Hours | nn | NN | 586 | 283 | 303 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 586 | 276 | 310 | 47.10% | 47.08% | 46.67% | 2.90 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 814 | 391 | 423 | 48.03% | 45.42% | 46.88% | 1.97 pp | -32 | 47 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 640 | 301 | 339 | 47.03% | 49.58% | 47.50% | 2.97 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 640 | 300 | 340 | 46.88% | 48.33% | 48.12% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 640 | 300 | 340 | 46.88% | 48.75% | 47.50% | 3.12 pp | -40 | 54 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 992 | 471 | 521 | 47.48% | 49.17% | 46.25% | 2.52 pp | -50 | 51 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 814 | 379 | 435 | 46.56% | 45.00% | 45.42% | 3.44 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 814 | 377 | 437 | 46.31% | 40.00% | 46.04% | 3.69 pp | -60 | 47 | -1.28 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 992 | 460 | 532 | 46.37% | 44.17% | 43.96% | 3.63 pp | -72 | 51 | -1.41 |
| BTC Market Hours | rf | RandomForest | 586 | 251 | 335 | 42.83% | 43.33% | 43.12% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 586 | 251 | 335 | 42.83% | 44.17% | 43.33% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | lstm | LSTM | 586 | 250 | 336 | 42.66% | 42.08% | 42.71% | 7.34 pp | -86 | 54 | -1.59 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 640 | 266 | 374 | 41.56% | 42.50% | 41.04% | 8.44 pp | -108 | 54 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 640 | 264 | 376 | 41.25% | 43.33% | 40.62% | 8.75 pp | -112 | 54 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 640 | 263 | 377 | 41.09% | 42.08% | 41.04% | 8.91 pp | -114 | 54 | -2.11 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 992 | 438 | 554 | 44.15% | 42.50% | 42.08% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 992 | 438 | 554 | 44.15% | 41.25% | 42.92% | 5.85 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| BTC Daily | lstm | LSTM | 814 | 345 | 469 | 42.38% | 36.25% | 40.83% | 7.62 pp | -124 | 47 | -2.64 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 814 | 337 | 477 | 41.40% | 37.50% | 40.83% | 8.60 pp | -140 | 47 | -2.98 |
| BTC Hourly | lstm | LSTM | 992 | 420 | 572 | 42.34% | 36.25% | 40.00% | 7.66 pp | -152 | 51 | -2.98 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 992 | 407 | 585 | 41.03% | 34.58% | 38.33% | 8.97 pp | -178 | 51 | -3.49 |
| BTC Daily | xgb | XGBoost | 824 | 323 | 501 | 39.20% | 36.67% | 36.04% | 10.80 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 992 | 471 | 521 | 47.48% | 49.17% | 46.25% | 2.52 pp | -50 | 51 | -0.98 |
| BTC Hourly | transformer | Transformer | 992 | 460 | 532 | 46.37% | 44.17% | 43.96% | 3.63 pp | -72 | 51 | -1.41 |
| BTC Hourly | nn | NN | 992 | 438 | 554 | 44.15% | 42.50% | 42.08% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 992 | 438 | 554 | 44.15% | 41.25% | 42.92% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 992 | 420 | 572 | 42.34% | 36.25% | 40.00% | 7.66 pp | -152 | 51 | -2.98 |
| BTC Hourly | xgb | XGBoost | 992 | 407 | 585 | 41.03% | 34.58% | 38.33% | 8.97 pp | -178 | 51 | -3.49 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 814 | 391 | 423 | 48.03% | 45.42% | 46.88% | 1.97 pp | -32 | 47 | -0.68 |
| BTC Daily | nn | NN | 814 | 379 | 435 | 46.56% | 45.00% | 45.42% | 3.44 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 814 | 377 | 437 | 46.31% | 40.00% | 46.04% | 3.69 pp | -60 | 47 | -1.28 |
| BTC Daily | lstm | LSTM | 814 | 345 | 469 | 42.38% | 36.25% | 40.83% | 7.62 pp | -124 | 47 | -2.64 |
| BTC Daily | rf | RandomForest | 814 | 337 | 477 | 41.40% | 37.50% | 40.83% | 8.60 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 824 | 323 | 501 | 39.20% | 36.67% | 36.04% | 10.80 pp | -178 | 47 | -3.79 |

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
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
