# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T14:44:45.315511+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1213 | 925 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1088 | 723 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 13:00:00+00:00 | 745 | 485 | 259 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 13:00:00+00:00 | 747 | 539 | 206 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 713 | 365 | 348 | 51.19% | 47.92% | 51.46% | 1.19 pp | 17 | 43 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 485 | 233 | 252 | 48.04% | 43.75% | 48.12% | 1.96 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 485 | 229 | 256 | 47.22% | 48.75% | 47.50% | 2.78 pp | -27 | 47 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 485 | 226 | 259 | 46.60% | 42.08% | 46.88% | 3.40 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 713 | 339 | 374 | 47.55% | 46.67% | 48.54% | 2.45 pp | -35 | 43 | -0.81 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 539 | 249 | 290 | 46.20% | 49.17% | 47.29% | 3.80 pp | -41 | 47 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 891 | 424 | 467 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 891 | 422 | 469 | 47.36% | 48.33% | 47.29% | 2.64 pp | -47 | 47 | -1.00 |
| BTC Daily | transformer | Transformer | 713 | 335 | 378 | 46.98% | 45.42% | 48.96% | 3.02 pp | -43 | 43 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 539 | 246 | 293 | 45.64% | 47.50% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| BTC Market Hours Daily | nn | NN | 539 | 246 | 293 | 45.64% | 44.17% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 485 | 209 | 276 | 43.09% | 41.67% | 43.12% | 6.91 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 485 | 207 | 278 | 42.68% | 42.08% | 42.92% | 7.32 pp | -71 | 47 | -1.51 |
| BTC Daily | lstm | LSTM | 713 | 322 | 391 | 45.16% | 38.75% | 44.58% | 4.84 pp | -69 | 43 | -1.60 |
| BTC Daily | rf | RandomForest | 713 | 318 | 395 | 44.60% | 41.67% | 44.58% | 5.40 pp | -77 | 43 | -1.79 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Hourly | nn | NN | 891 | 399 | 492 | 44.78% | 46.25% | 42.92% | 5.22 pp | -93 | 47 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 485 | 196 | 289 | 40.41% | 39.58% | 40.62% | 9.59 pp | -93 | 47 | -1.98 |
| BTC Hourly | rf | RandomForest | 891 | 398 | 493 | 44.67% | 45.42% | 44.17% | 5.33 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 539 | 221 | 318 | 41.00% | 41.25% | 41.25% | 9.00 pp | -97 | 47 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 539 | 216 | 323 | 40.07% | 37.92% | 40.62% | 9.93 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 539 | 214 | 325 | 39.70% | 39.58% | 39.38% | 10.30 pp | -111 | 47 | -2.36 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 891 | 382 | 509 | 42.87% | 39.17% | 42.29% | 7.13 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 891 | 377 | 514 | 42.31% | 42.92% | 42.29% | 7.69 pp | -137 | 47 | -2.91 |
| BTC Daily | xgb | XGBoost | 723 | 290 | 433 | 40.11% | 36.25% | 39.38% | 9.89 pp | -143 | 43 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 891 | 424 | 467 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 891 | 422 | 469 | 47.36% | 48.33% | 47.29% | 2.64 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 891 | 399 | 492 | 44.78% | 46.25% | 42.92% | 5.22 pp | -93 | 47 | -1.98 |
| BTC Hourly | rf | RandomForest | 891 | 398 | 493 | 44.67% | 45.42% | 44.17% | 5.33 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 891 | 382 | 509 | 42.87% | 39.17% | 42.29% | 7.13 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 891 | 377 | 514 | 42.31% | 42.92% | 42.29% | 7.69 pp | -137 | 47 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 713 | 365 | 348 | 51.19% | 47.92% | 51.46% | 1.19 pp | 17 | 43 | 0.40 |
| BTC Daily | nn | NN | 713 | 339 | 374 | 47.55% | 46.67% | 48.54% | 2.45 pp | -35 | 43 | -0.81 |
| BTC Daily | transformer | Transformer | 713 | 335 | 378 | 46.98% | 45.42% | 48.96% | 3.02 pp | -43 | 43 | -1.00 |
| BTC Daily | lstm | LSTM | 713 | 322 | 391 | 45.16% | 38.75% | 44.58% | 4.84 pp | -69 | 43 | -1.60 |
| BTC Daily | rf | RandomForest | 713 | 318 | 395 | 44.60% | 41.67% | 44.58% | 5.40 pp | -77 | 43 | -1.79 |
| BTC Daily | xgb | XGBoost | 723 | 290 | 433 | 40.11% | 36.25% | 39.38% | 9.89 pp | -143 | 43 | -3.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 485 | 233 | 252 | 48.04% | 43.75% | 48.12% | 1.96 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 485 | 229 | 256 | 47.22% | 48.75% | 47.50% | 2.78 pp | -27 | 47 | -0.57 |
| BTC Market Hours | transformer | Transformer | 485 | 226 | 259 | 46.60% | 42.08% | 46.88% | 3.40 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 485 | 209 | 276 | 43.09% | 41.67% | 43.12% | 6.91 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 485 | 207 | 278 | 42.68% | 42.08% | 42.92% | 7.32 pp | -71 | 47 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 485 | 196 | 289 | 40.41% | 39.58% | 40.62% | 9.59 pp | -93 | 47 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 539 | 249 | 290 | 46.20% | 49.17% | 47.29% | 3.80 pp | -41 | 47 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 539 | 246 | 293 | 45.64% | 47.50% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| BTC Market Hours Daily | nn | NN | 539 | 246 | 293 | 45.64% | 44.17% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 539 | 221 | 318 | 41.00% | 41.25% | 41.25% | 9.00 pp | -97 | 47 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 539 | 216 | 323 | 40.07% | 37.92% | 40.62% | 9.93 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 539 | 214 | 325 | 39.70% | 39.58% | 39.38% | 10.30 pp | -111 | 47 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
