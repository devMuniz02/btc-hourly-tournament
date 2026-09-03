# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T13:47:33.654665+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1212 | 924 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1088 | 723 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 744 | 485 | 258 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 746 | 539 | 205 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 134 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 713 | 366 | 347 | 51.33% | 48.33% | 51.67% | 1.33 pp | 19 | 43 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 485 | 233 | 252 | 48.04% | 43.75% | 48.12% | 1.96 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 485 | 229 | 256 | 47.22% | 48.75% | 47.50% | 2.78 pp | -27 | 47 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 485 | 226 | 259 | 46.60% | 42.08% | 46.88% | 3.40 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 713 | 340 | 373 | 47.69% | 47.08% | 48.75% | 2.31 pp | -33 | 43 | -0.77 |
| Consolidated Hourly | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 539 | 249 | 290 | 46.20% | 49.17% | 47.29% | 3.80 pp | -41 | 47 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 890 | 424 | 466 | 47.64% | 49.58% | 48.12% | 2.36 pp | -42 | 47 | -0.89 |
| BTC Daily | transformer | Transformer | 713 | 336 | 377 | 47.12% | 45.83% | 49.17% | 2.88 pp | -41 | 43 | -0.95 |
| BTC Hourly | transformer | Transformer | 890 | 422 | 468 | 47.42% | 48.75% | 47.50% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 539 | 246 | 293 | 45.64% | 47.50% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| BTC Market Hours Daily | nn | NN | 539 | 246 | 293 | 45.64% | 44.17% | 46.67% | 4.36 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 485 | 209 | 276 | 43.09% | 41.67% | 43.12% | 6.91 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 485 | 207 | 278 | 42.68% | 42.08% | 42.92% | 7.32 pp | -71 | 47 | -1.51 |
| Consolidated Hourly | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| BTC Daily | lstm | LSTM | 713 | 321 | 392 | 45.02% | 38.33% | 44.38% | 4.98 pp | -71 | 43 | -1.65 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 713 | 319 | 394 | 44.74% | 42.08% | 44.79% | 5.26 pp | -75 | 43 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 485 | 196 | 289 | 40.41% | 39.58% | 40.62% | 9.59 pp | -93 | 47 | -1.98 |
| BTC Hourly | nn | NN | 890 | 398 | 492 | 44.72% | 45.83% | 42.92% | 5.28 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 890 | 397 | 493 | 44.61% | 45.00% | 44.17% | 5.39 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 539 | 221 | 318 | 41.00% | 41.25% | 41.25% | 9.00 pp | -97 | 47 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 539 | 216 | 323 | 40.07% | 37.92% | 40.62% | 9.93 pp | -107 | 47 | -2.28 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 539 | 214 | 325 | 39.70% | 39.58% | 39.38% | 10.30 pp | -111 | 47 | -2.36 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 890 | 381 | 509 | 42.81% | 38.75% | 42.08% | 7.19 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 890 | 376 | 514 | 42.25% | 42.50% | 42.29% | 7.75 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 723 | 291 | 432 | 40.25% | 36.67% | 39.58% | 9.75 pp | -141 | 43 | -3.28 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 890 | 424 | 466 | 47.64% | 49.58% | 48.12% | 2.36 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 890 | 422 | 468 | 47.42% | 48.75% | 47.50% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 890 | 398 | 492 | 44.72% | 45.83% | 42.92% | 5.28 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 890 | 397 | 493 | 44.61% | 45.00% | 44.17% | 5.39 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 890 | 381 | 509 | 42.81% | 38.75% | 42.08% | 7.19 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 890 | 376 | 514 | 42.25% | 42.50% | 42.29% | 7.75 pp | -138 | 47 | -2.94 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 713 | 366 | 347 | 51.33% | 48.33% | 51.67% | 1.33 pp | 19 | 43 | 0.44 |
| BTC Daily | nn | NN | 713 | 340 | 373 | 47.69% | 47.08% | 48.75% | 2.31 pp | -33 | 43 | -0.77 |
| BTC Daily | transformer | Transformer | 713 | 336 | 377 | 47.12% | 45.83% | 49.17% | 2.88 pp | -41 | 43 | -0.95 |
| BTC Daily | lstm | LSTM | 713 | 321 | 392 | 45.02% | 38.33% | 44.38% | 4.98 pp | -71 | 43 | -1.65 |
| BTC Daily | rf | RandomForest | 713 | 319 | 394 | 44.74% | 42.08% | 44.79% | 5.26 pp | -75 | 43 | -1.74 |
| BTC Daily | xgb | XGBoost | 723 | 291 | 432 | 40.25% | 36.67% | 39.58% | 9.75 pp | -141 | 43 | -3.28 |

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
| Consolidated Hourly | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
