# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T19:21:24.709842+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 18:00:00+00:00 | 930 | 639 | 289 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 586 | 285 | 301 | 48.63% | 47.92% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 586 | 283 | 303 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 586 | 276 | 310 | 47.10% | 47.08% | 46.67% | 2.90 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 814 | 392 | 422 | 48.16% | 45.83% | 47.08% | 1.84 pp | -30 | 47 | -0.64 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 639 | 300 | 339 | 46.95% | 49.17% | 47.50% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 639 | 300 | 339 | 46.95% | 48.33% | 48.33% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 639 | 299 | 340 | 46.79% | 48.75% | 47.29% | 3.21 pp | -41 | 54 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 991 | 471 | 520 | 47.53% | 49.58% | 46.46% | 2.47 pp | -49 | 51 | -0.96 |
| BTC Daily | nn | NN | 814 | 379 | 435 | 46.56% | 45.00% | 45.42% | 3.44 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 814 | 378 | 436 | 46.44% | 40.00% | 46.25% | 3.56 pp | -58 | 47 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 991 | 460 | 531 | 46.42% | 44.58% | 43.96% | 3.58 pp | -71 | 51 | -1.39 |
| BTC Market Hours | rf | RandomForest | 586 | 251 | 335 | 42.83% | 43.33% | 43.12% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 586 | 251 | 335 | 42.83% | 44.17% | 43.33% | 7.17 pp | -84 | 54 | -1.56 |
| BTC Market Hours | lstm | LSTM | 586 | 250 | 336 | 42.66% | 42.08% | 42.71% | 7.34 pp | -86 | 54 | -1.59 |
| BTC Market Hours Daily | rf | RandomForest | 639 | 266 | 373 | 41.63% | 42.50% | 41.04% | 8.37 pp | -107 | 54 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 639 | 263 | 376 | 41.16% | 43.33% | 40.42% | 8.84 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 639 | 262 | 377 | 41.00% | 41.67% | 40.83% | 9.00 pp | -115 | 54 | -2.13 |
| Consolidated Hourly | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| BTC Hourly | nn | NN | 991 | 437 | 554 | 44.10% | 42.08% | 42.08% | 5.90 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 991 | 437 | 554 | 44.10% | 41.25% | 42.71% | 5.90 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 814 | 344 | 470 | 42.26% | 35.83% | 40.62% | 7.74 pp | -126 | 47 | -2.68 |
| BTC Hourly | lstm | LSTM | 991 | 420 | 571 | 42.38% | 36.67% | 40.00% | 7.62 pp | -151 | 51 | -2.96 |
| BTC Daily | rf | RandomForest | 814 | 337 | 477 | 41.40% | 37.08% | 40.83% | 8.60 pp | -140 | 47 | -2.98 |
| Consolidated Hourly | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 991 | 407 | 584 | 41.07% | 35.00% | 38.33% | 8.93 pp | -177 | 51 | -3.47 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 639 | 300 | 339 | 46.95% | 49.17% | 47.50% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 639 | 300 | 339 | 46.95% | 48.33% | 48.33% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 639 | 299 | 340 | 46.79% | 48.75% | 47.29% | 3.21 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 639 | 266 | 373 | 41.63% | 42.50% | 41.04% | 8.37 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 639 | 263 | 376 | 41.16% | 43.33% | 40.42% | 8.84 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 639 | 262 | 377 | 41.00% | 41.67% | 40.83% | 9.00 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
