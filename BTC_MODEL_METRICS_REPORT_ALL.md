# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T06:54:39.461601+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1140 | 852 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1016 | 651 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 619 | 413 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 621 | 467 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 1 | 69 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 20:00:00+00:00 | 70 | 1 | 69 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 413 | 205 | 208 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 641 | 312 | 329 | 48.67% | 46.25% | 49.38% | 1.33 pp | -17 | 40 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 641 | 311 | 330 | 48.52% | 45.42% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 413 | 195 | 218 | 47.22% | 50.42% | 47.22% | 2.78 pp | -23 | 41 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 467 | 217 | 250 | 46.47% | 46.67% | 46.47% | 3.53 pp | -33 | 41 | -0.80 |
| BTC Market Hours | transformer | Transformer | 413 | 190 | 223 | 46.00% | 41.67% | 46.00% | 4.00 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 467 | 214 | 253 | 45.82% | 45.83% | 45.82% | 4.18 pp | -39 | 41 | -0.95 |
| BTC Daily | nn | NN | 641 | 301 | 340 | 46.96% | 42.50% | 48.96% | 3.04 pp | -39 | 40 | -0.97 |
| BTC Hourly | transformer | Transformer | 818 | 387 | 431 | 47.31% | 46.25% | 46.46% | 2.69 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | nn | NN | 467 | 213 | 254 | 45.61% | 45.00% | 45.61% | 4.39 pp | -41 | 41 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 818 | 385 | 433 | 47.07% | 43.75% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 413 | 182 | 231 | 44.07% | 44.58% | 44.07% | 5.93 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 413 | 178 | 235 | 43.10% | 42.08% | 43.10% | 6.90 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| BTC Hourly | nn | NN | 818 | 370 | 448 | 45.23% | 42.08% | 45.00% | 4.77 pp | -78 | 44 | -1.77 |
| BTC Daily | lstm | LSTM | 641 | 284 | 357 | 44.31% | 41.67% | 43.96% | 5.69 pp | -73 | 40 | -1.82 |
| BTC Hourly | rf | RandomForest | 818 | 366 | 452 | 44.74% | 45.00% | 44.58% | 5.26 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 467 | 192 | 275 | 41.11% | 41.67% | 41.11% | 8.89 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 467 | 191 | 276 | 40.90% | 40.00% | 40.90% | 9.10 pp | -85 | 41 | -2.07 |
| BTC Market Hours | xgb | XGBoost | 413 | 164 | 249 | 39.71% | 37.50% | 39.71% | 10.29 pp | -85 | 41 | -2.07 |
| BTC Hourly | lstm | LSTM | 818 | 356 | 462 | 43.52% | 42.08% | 43.96% | 6.48 pp | -106 | 44 | -2.41 |
| BTC Daily | rf | RandomForest | 641 | 272 | 369 | 42.43% | 40.83% | 43.33% | 7.57 pp | -97 | 40 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 467 | 181 | 286 | 38.76% | 35.42% | 38.76% | 11.24 pp | -105 | 41 | -2.56 |
| BTC Hourly | xgb | XGBoost | 818 | 347 | 471 | 42.42% | 40.42% | 42.92% | 7.58 pp | -124 | 44 | -2.82 |
| Consolidated Hourly | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |
| BTC Daily | xgb | XGBoost | 651 | 254 | 397 | 39.02% | 30.42% | 38.96% | 10.98 pp | -143 | 40 | -3.58 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 818 | 387 | 431 | 47.31% | 46.25% | 46.46% | 2.69 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 818 | 385 | 433 | 47.07% | 43.75% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 818 | 370 | 448 | 45.23% | 42.08% | 45.00% | 4.77 pp | -78 | 44 | -1.77 |
| BTC Hourly | rf | RandomForest | 818 | 366 | 452 | 44.74% | 45.00% | 44.58% | 5.26 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 818 | 356 | 462 | 43.52% | 42.08% | 43.96% | 6.48 pp | -106 | 44 | -2.41 |
| BTC Hourly | xgb | XGBoost | 818 | 347 | 471 | 42.42% | 40.42% | 42.92% | 7.58 pp | -124 | 44 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 641 | 312 | 329 | 48.67% | 46.25% | 49.38% | 1.33 pp | -17 | 40 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 641 | 311 | 330 | 48.52% | 45.42% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 641 | 301 | 340 | 46.96% | 42.50% | 48.96% | 3.04 pp | -39 | 40 | -0.97 |
| BTC Daily | lstm | LSTM | 641 | 284 | 357 | 44.31% | 41.67% | 43.96% | 5.69 pp | -73 | 40 | -1.82 |
| BTC Daily | rf | RandomForest | 641 | 272 | 369 | 42.43% | 40.83% | 43.33% | 7.57 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 651 | 254 | 397 | 39.02% | 30.42% | 38.96% | 10.98 pp | -143 | 40 | -3.58 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 413 | 205 | 208 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 413 | 195 | 218 | 47.22% | 50.42% | 47.22% | 2.78 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 413 | 190 | 223 | 46.00% | 41.67% | 46.00% | 4.00 pp | -33 | 41 | -0.80 |
| BTC Market Hours | lstm | LSTM | 413 | 182 | 231 | 44.07% | 44.58% | 44.07% | 5.93 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 413 | 178 | 235 | 43.10% | 42.08% | 43.10% | 6.90 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 413 | 164 | 249 | 39.71% | 37.50% | 39.71% | 10.29 pp | -85 | 41 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 467 | 217 | 250 | 46.47% | 46.67% | 46.47% | 3.53 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 467 | 214 | 253 | 45.82% | 45.83% | 45.82% | 4.18 pp | -39 | 41 | -0.95 |
| BTC Market Hours Daily | nn | NN | 467 | 213 | 254 | 45.61% | 45.00% | 45.61% | 4.39 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 467 | 192 | 275 | 41.11% | 41.67% | 41.11% | 8.89 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 467 | 191 | 276 | 40.90% | 40.00% | 40.90% | 9.10 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 467 | 181 | 286 | 38.76% | 35.42% | 38.76% | 11.24 pp | -105 | 41 | -2.56 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 25 | 45 | 35.71% | 35.71% | 35.71% | 14.29 pp | -20 | 7 | -2.86 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
