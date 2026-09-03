# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T02:28:48.005573+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1204 | 916 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1080 | 715 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 735 | 477 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 737 | 531 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T21:00:00+00:00 | 127 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T21:00:00+00:00 | 127 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T21:00:00+00:00 | 127 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T21:00:00+00:00 | 128 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 127 | 65 | 62 | 51.18% | 51.18% | 51.18% | 1.18 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 65 | 62 | 51.18% | 51.18% | 51.18% | 1.18 pp | 3 | 10 | 0.30 |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 705 | 345 | 360 | 48.94% | 47.50% | 48.96% | 1.06 pp | -15 | 42 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 477 | 230 | 247 | 48.22% | 43.75% | 48.22% | 1.78 pp | -17 | 46 | -0.37 |
| Consolidated Hourly | xgb | XGBoost | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 477 | 224 | 253 | 46.96% | 47.92% | 46.96% | 3.04 pp | -29 | 46 | -0.63 |
| BTC Daily | transformer | Transformer | 705 | 339 | 366 | 48.09% | 47.50% | 50.21% | 1.91 pp | -27 | 42 | -0.64 |
| BTC Market Hours | transformer | Transformer | 477 | 222 | 255 | 46.54% | 41.25% | 46.54% | 3.46 pp | -33 | 46 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 531 | 245 | 286 | 46.14% | 48.75% | 46.88% | 3.86 pp | -41 | 46 | -0.89 |
| Consolidated Hourly | lstm | LSTM | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 531 | 243 | 288 | 45.76% | 47.50% | 46.46% | 4.24 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 531 | 243 | 288 | 45.76% | 43.33% | 46.46% | 4.24 pp | -45 | 46 | -0.98 |
| BTC Hourly | transformer | Transformer | 882 | 418 | 464 | 47.39% | 48.75% | 47.71% | 2.61 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 882 | 417 | 465 | 47.28% | 47.92% | 47.50% | 2.72 pp | -48 | 47 | -1.02 |
| BTC Daily | nn | NN | 705 | 327 | 378 | 46.38% | 43.33% | 48.33% | 3.62 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 477 | 206 | 271 | 43.19% | 41.67% | 43.19% | 6.81 pp | -65 | 46 | -1.41 |
| Consolidated Hourly | nn | NN | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| BTC Market Hours | rf | RandomForest | 477 | 204 | 273 | 42.77% | 42.08% | 42.77% | 7.23 pp | -69 | 46 | -1.50 |
| BTC Hourly | nn | NN | 882 | 396 | 486 | 44.90% | 45.83% | 43.54% | 5.10 pp | -90 | 47 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 477 | 194 | 283 | 40.67% | 39.58% | 40.67% | 9.33 pp | -89 | 46 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 531 | 219 | 312 | 41.24% | 41.25% | 41.25% | 8.76 pp | -93 | 46 | -2.02 |
| BTC Hourly | rf | RandomForest | 882 | 392 | 490 | 44.44% | 44.17% | 43.96% | 5.56 pp | -98 | 47 | -2.09 |
| BTC Daily | lstm | LSTM | 705 | 306 | 399 | 43.40% | 38.75% | 42.50% | 6.60 pp | -93 | 42 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 531 | 212 | 319 | 39.92% | 38.75% | 39.17% | 10.08 pp | -107 | 46 | -2.33 |
| BTC Daily | rf | RandomForest | 705 | 303 | 402 | 42.98% | 41.67% | 43.54% | 7.02 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 531 | 211 | 320 | 39.74% | 37.08% | 40.62% | 10.26 pp | -109 | 46 | -2.37 |
| BTC Hourly | lstm | LSTM | 882 | 376 | 506 | 42.63% | 37.92% | 41.88% | 7.37 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 882 | 373 | 509 | 42.29% | 41.67% | 42.50% | 7.71 pp | -136 | 47 | -2.89 |
| Consolidated Market Hours Daily | nn | NN | 25 | 9 | 16 | 36.00% | 36.00% | 36.00% | 14.00 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 715 | 282 | 433 | 39.44% | 34.58% | 39.38% | 10.56 pp | -151 | 42 | -3.60 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 882 | 418 | 464 | 47.39% | 48.75% | 47.71% | 2.61 pp | -46 | 47 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 882 | 417 | 465 | 47.28% | 47.92% | 47.50% | 2.72 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 882 | 396 | 486 | 44.90% | 45.83% | 43.54% | 5.10 pp | -90 | 47 | -1.91 |
| BTC Hourly | rf | RandomForest | 882 | 392 | 490 | 44.44% | 44.17% | 43.96% | 5.56 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 882 | 376 | 506 | 42.63% | 37.92% | 41.88% | 7.37 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 882 | 373 | 509 | 42.29% | 41.67% | 42.50% | 7.71 pp | -136 | 47 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 705 | 345 | 360 | 48.94% | 47.50% | 48.96% | 1.06 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 705 | 339 | 366 | 48.09% | 47.50% | 50.21% | 1.91 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 705 | 327 | 378 | 46.38% | 43.33% | 48.33% | 3.62 pp | -51 | 42 | -1.21 |
| BTC Daily | lstm | LSTM | 705 | 306 | 399 | 43.40% | 38.75% | 42.50% | 6.60 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 705 | 303 | 402 | 42.98% | 41.67% | 43.54% | 7.02 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 715 | 282 | 433 | 39.44% | 34.58% | 39.38% | 10.56 pp | -151 | 42 | -3.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 477 | 230 | 247 | 48.22% | 43.75% | 48.22% | 1.78 pp | -17 | 46 | -0.37 |
| BTC Market Hours | nn | NN | 477 | 224 | 253 | 46.96% | 47.92% | 46.96% | 3.04 pp | -29 | 46 | -0.63 |
| BTC Market Hours | transformer | Transformer | 477 | 222 | 255 | 46.54% | 41.25% | 46.54% | 3.46 pp | -33 | 46 | -0.72 |
| BTC Market Hours | lstm | LSTM | 477 | 206 | 271 | 43.19% | 41.67% | 43.19% | 6.81 pp | -65 | 46 | -1.41 |
| BTC Market Hours | rf | RandomForest | 477 | 204 | 273 | 42.77% | 42.08% | 42.77% | 7.23 pp | -69 | 46 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 477 | 194 | 283 | 40.67% | 39.58% | 40.67% | 9.33 pp | -89 | 46 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 531 | 245 | 286 | 46.14% | 48.75% | 46.88% | 3.86 pp | -41 | 46 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 531 | 243 | 288 | 45.76% | 47.50% | 46.46% | 4.24 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 531 | 243 | 288 | 45.76% | 43.33% | 46.46% | 4.24 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 531 | 219 | 312 | 41.24% | 41.25% | 41.25% | 8.76 pp | -93 | 46 | -2.02 |
| BTC Market Hours Daily | xgb | XGBoost | 531 | 212 | 319 | 39.92% | 38.75% | 39.17% | 10.08 pp | -107 | 46 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 531 | 211 | 320 | 39.74% | 37.08% | 40.62% | 10.26 pp | -109 | 46 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 127 | 65 | 62 | 51.18% | 51.18% | 51.18% | 1.18 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | xgb | XGBoost | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | nn | NN | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 65 | 62 | 51.18% | 51.18% | 51.18% | 1.18 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 56 | 71 | 44.09% | 44.09% | 44.09% | 5.91 pp | -15 | 10 | -1.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 9 | 16 | 36.00% | 36.00% | 36.00% | 14.00 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
