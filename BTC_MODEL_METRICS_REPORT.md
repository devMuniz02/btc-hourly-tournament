# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T22:45:52.268289+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1315 | 1027 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1191 | 826 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 934 | 588 | 345 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 935 | 641 | 292 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 588 | 285 | 303 | 48.47% | 47.92% | 47.50% | 1.53 pp | -18 | 55 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| BTC Market Hours | nn | NN | 588 | 284 | 304 | 48.30% | 52.50% | 50.00% | 1.70 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 588 | 277 | 311 | 47.11% | 47.08% | 46.46% | 2.89 pp | -34 | 55 | -0.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 816 | 392 | 424 | 48.04% | 45.42% | 46.88% | 1.96 pp | -32 | 47 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 641 | 301 | 340 | 46.96% | 49.58% | 47.50% | 3.04 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 641 | 300 | 341 | 46.80% | 48.33% | 48.12% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 641 | 300 | 341 | 46.80% | 48.75% | 47.50% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 993 | 471 | 522 | 47.43% | 49.17% | 46.25% | 2.57 pp | -51 | 51 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 816 | 381 | 435 | 46.69% | 45.42% | 45.42% | 3.31 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 816 | 379 | 437 | 46.45% | 40.00% | 46.25% | 3.55 pp | -58 | 47 | -1.23 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 993 | 461 | 532 | 46.42% | 44.58% | 44.17% | 3.58 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 588 | 251 | 337 | 42.69% | 42.50% | 42.50% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 588 | 251 | 337 | 42.69% | 43.33% | 42.92% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 588 | 251 | 337 | 42.69% | 44.17% | 43.12% | 7.31 pp | -86 | 55 | -1.56 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 641 | 266 | 375 | 41.50% | 42.50% | 41.04% | 8.50 pp | -109 | 54 | -2.02 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 641 | 264 | 377 | 41.19% | 42.92% | 40.62% | 8.81 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 641 | 263 | 378 | 41.03% | 42.08% | 41.04% | 8.97 pp | -115 | 54 | -2.13 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 993 | 438 | 555 | 44.11% | 42.50% | 42.08% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 993 | 438 | 555 | 44.11% | 41.25% | 42.92% | 5.89 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| BTC Daily | lstm | LSTM | 816 | 345 | 471 | 42.28% | 35.83% | 40.62% | 7.72 pp | -126 | 47 | -2.68 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| BTC Hourly | lstm | LSTM | 993 | 421 | 572 | 42.40% | 36.67% | 40.00% | 7.60 pp | -151 | 51 | -2.96 |
| BTC Daily | rf | RandomForest | 816 | 338 | 478 | 41.42% | 37.08% | 40.62% | 8.58 pp | -140 | 47 | -2.98 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 993 | 407 | 586 | 40.99% | 34.58% | 38.33% | 9.01 pp | -179 | 51 | -3.51 |
| BTC Daily | xgb | XGBoost | 826 | 324 | 502 | 39.23% | 36.67% | 35.83% | 10.77 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 993 | 471 | 522 | 47.43% | 49.17% | 46.25% | 2.57 pp | -51 | 51 | -1.00 |
| BTC Hourly | transformer | Transformer | 993 | 461 | 532 | 46.42% | 44.58% | 44.17% | 3.58 pp | -71 | 51 | -1.39 |
| BTC Hourly | nn | NN | 993 | 438 | 555 | 44.11% | 42.50% | 42.08% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 993 | 438 | 555 | 44.11% | 41.25% | 42.92% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 993 | 421 | 572 | 42.40% | 36.67% | 40.00% | 7.60 pp | -151 | 51 | -2.96 |
| BTC Hourly | xgb | XGBoost | 993 | 407 | 586 | 40.99% | 34.58% | 38.33% | 9.01 pp | -179 | 51 | -3.51 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 816 | 392 | 424 | 48.04% | 45.42% | 46.88% | 1.96 pp | -32 | 47 | -0.68 |
| BTC Daily | nn | NN | 816 | 381 | 435 | 46.69% | 45.42% | 45.42% | 3.31 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 816 | 379 | 437 | 46.45% | 40.00% | 46.25% | 3.55 pp | -58 | 47 | -1.23 |
| BTC Daily | lstm | LSTM | 816 | 345 | 471 | 42.28% | 35.83% | 40.62% | 7.72 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 816 | 338 | 478 | 41.42% | 37.08% | 40.62% | 8.58 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 826 | 324 | 502 | 39.23% | 36.67% | 35.83% | 10.77 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 588 | 285 | 303 | 48.47% | 47.92% | 47.50% | 1.53 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 588 | 284 | 304 | 48.30% | 52.50% | 50.00% | 1.70 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 588 | 277 | 311 | 47.11% | 47.08% | 46.46% | 2.89 pp | -34 | 55 | -0.62 |
| BTC Market Hours | lstm | LSTM | 588 | 251 | 337 | 42.69% | 42.50% | 42.50% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 588 | 251 | 337 | 42.69% | 43.33% | 42.92% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 588 | 251 | 337 | 42.69% | 44.17% | 43.12% | 7.31 pp | -86 | 55 | -1.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 641 | 301 | 340 | 46.96% | 49.58% | 47.50% | 3.04 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 641 | 300 | 341 | 46.80% | 48.33% | 48.12% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 641 | 300 | 341 | 46.80% | 48.75% | 47.50% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 641 | 266 | 375 | 41.50% | 42.50% | 41.04% | 8.50 pp | -109 | 54 | -2.02 |
| BTC Market Hours Daily | xgb | XGBoost | 641 | 264 | 377 | 41.19% | 42.92% | 40.62% | 8.81 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 641 | 263 | 378 | 41.03% | 42.08% | 41.04% | 8.97 pp | -115 | 54 | -2.13 |

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
