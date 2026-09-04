# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T09:44:05.236505+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1225 | 937 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1101 | 736 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 769 | 498 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 771 | 552 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 148 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 498 | 242 | 256 | 48.59% | 45.42% | 48.33% | 1.41 pp | -14 | 48 | -0.29 |
| Consolidated Hourly | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 498 | 236 | 262 | 47.39% | 50.83% | 47.92% | 2.61 pp | -26 | 48 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 726 | 350 | 376 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 498 | 234 | 264 | 46.99% | 44.58% | 47.71% | 3.01 pp | -30 | 48 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| BTC Daily | transformer | Transformer | 726 | 346 | 380 | 47.66% | 46.67% | 49.58% | 2.34 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 552 | 257 | 295 | 46.56% | 49.17% | 47.50% | 3.44 pp | -38 | 48 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 552 | 255 | 297 | 46.20% | 49.58% | 46.88% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 552 | 255 | 297 | 46.20% | 45.00% | 47.50% | 3.80 pp | -42 | 48 | -0.88 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Daily | nn | NN | 726 | 336 | 390 | 46.28% | 44.58% | 47.29% | 3.72 pp | -54 | 43 | -1.26 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 498 | 215 | 283 | 43.17% | 41.25% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 498 | 215 | 283 | 43.17% | 44.17% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| Consolidated Hourly | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 498 | 205 | 293 | 41.16% | 41.67% | 41.25% | 8.84 pp | -88 | 48 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 552 | 229 | 323 | 41.49% | 42.50% | 40.83% | 8.51 pp | -94 | 48 | -1.96 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 552 | 222 | 330 | 40.22% | 38.33% | 40.42% | 9.78 pp | -108 | 48 | -2.25 |
| Consolidated Hourly | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 552 | 220 | 332 | 39.86% | 41.25% | 38.96% | 10.14 pp | -112 | 48 | -2.33 |
| BTC Daily | lstm | LSTM | 726 | 312 | 414 | 42.98% | 36.67% | 41.25% | 7.02 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 726 | 310 | 416 | 42.70% | 40.83% | 43.33% | 7.30 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 736 | 292 | 444 | 39.67% | 37.08% | 38.54% | 10.33 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 726 | 350 | 376 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 726 | 346 | 380 | 47.66% | 46.67% | 49.58% | 2.34 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 726 | 336 | 390 | 46.28% | 44.58% | 47.29% | 3.72 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 726 | 312 | 414 | 42.98% | 36.67% | 41.25% | 7.02 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 726 | 310 | 416 | 42.70% | 40.83% | 43.33% | 7.30 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 736 | 292 | 444 | 39.67% | 37.08% | 38.54% | 10.33 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 498 | 242 | 256 | 48.59% | 45.42% | 48.33% | 1.41 pp | -14 | 48 | -0.29 |
| BTC Market Hours | nn | NN | 498 | 236 | 262 | 47.39% | 50.83% | 47.92% | 2.61 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 498 | 234 | 264 | 46.99% | 44.58% | 47.71% | 3.01 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 498 | 215 | 283 | 43.17% | 41.25% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 498 | 215 | 283 | 43.17% | 44.17% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 498 | 205 | 293 | 41.16% | 41.67% | 41.25% | 8.84 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 552 | 257 | 295 | 46.56% | 49.17% | 47.50% | 3.44 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 552 | 255 | 297 | 46.20% | 49.58% | 46.88% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 552 | 255 | 297 | 46.20% | 45.00% | 47.50% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 552 | 229 | 323 | 41.49% | 42.50% | 40.83% | 8.51 pp | -94 | 48 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 552 | 222 | 330 | 40.22% | 38.33% | 40.42% | 9.78 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 552 | 220 | 332 | 39.86% | 41.25% | 38.96% | 10.14 pp | -112 | 48 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
