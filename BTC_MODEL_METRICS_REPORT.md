# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T14:01:17.988287+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1261 | 973 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1137 | 772 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 13:00:00+00:00 | 833 | 534 | 298 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 13:00:00+00:00 | 835 | 588 | 245 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 534 | 259 | 275 | 48.50% | 45.42% | 48.54% | 1.50 pp | -16 | 50 | -0.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 534 | 256 | 278 | 47.94% | 48.75% | 48.54% | 2.06 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 762 | 370 | 392 | 48.56% | 47.92% | 48.54% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Market Hours | nn | NN | 534 | 253 | 281 | 47.38% | 50.42% | 48.96% | 2.62 pp | -28 | 50 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 588 | 279 | 309 | 47.45% | 50.83% | 48.54% | 2.55 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 588 | 275 | 313 | 46.77% | 46.67% | 48.12% | 3.23 pp | -38 | 50 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 588 | 274 | 314 | 46.60% | 51.25% | 47.50% | 3.40 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 939 | 448 | 491 | 47.71% | 50.00% | 46.88% | 2.29 pp | -43 | 49 | -0.88 |
| BTC Daily | transformer | Transformer | 762 | 359 | 403 | 47.11% | 42.92% | 47.92% | 2.89 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 939 | 441 | 498 | 46.96% | 46.67% | 45.42% | 3.04 pp | -57 | 49 | -1.16 |
| BTC Daily | nn | NN | 762 | 354 | 408 | 46.46% | 45.42% | 46.67% | 3.54 pp | -54 | 45 | -1.20 |
| BTC Market Hours | lstm | LSTM | 534 | 231 | 303 | 43.26% | 42.08% | 44.17% | 6.74 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 534 | 230 | 304 | 43.07% | 44.17% | 43.75% | 6.93 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 534 | 222 | 312 | 41.57% | 42.92% | 42.29% | 8.43 pp | -90 | 50 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 588 | 245 | 343 | 41.67% | 44.58% | 41.46% | 8.33 pp | -98 | 50 | -1.96 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 939 | 418 | 521 | 44.52% | 44.58% | 43.75% | 5.48 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 939 | 417 | 522 | 44.41% | 42.50% | 42.29% | 5.59 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 588 | 239 | 349 | 40.65% | 40.00% | 40.42% | 9.35 pp | -110 | 50 | -2.20 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 588 | 235 | 353 | 39.97% | 41.67% | 38.96% | 10.03 pp | -118 | 50 | -2.36 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 762 | 322 | 440 | 42.26% | 35.83% | 40.62% | 7.74 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 762 | 319 | 443 | 41.86% | 38.33% | 42.29% | 8.14 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 939 | 400 | 539 | 42.60% | 36.25% | 41.67% | 7.40 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 939 | 395 | 544 | 42.07% | 41.25% | 40.62% | 7.93 pp | -149 | 49 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 772 | 304 | 468 | 39.38% | 35.42% | 37.29% | 10.62 pp | -164 | 45 | -3.64 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 939 | 448 | 491 | 47.71% | 50.00% | 46.88% | 2.29 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 939 | 441 | 498 | 46.96% | 46.67% | 45.42% | 3.04 pp | -57 | 49 | -1.16 |
| BTC Hourly | rf | RandomForest | 939 | 418 | 521 | 44.52% | 44.58% | 43.75% | 5.48 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 939 | 417 | 522 | 44.41% | 42.50% | 42.29% | 5.59 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 939 | 400 | 539 | 42.60% | 36.25% | 41.67% | 7.40 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 939 | 395 | 544 | 42.07% | 41.25% | 40.62% | 7.93 pp | -149 | 49 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 762 | 370 | 392 | 48.56% | 47.92% | 48.54% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Daily | transformer | Transformer | 762 | 359 | 403 | 47.11% | 42.92% | 47.92% | 2.89 pp | -44 | 45 | -0.98 |
| BTC Daily | nn | NN | 762 | 354 | 408 | 46.46% | 45.42% | 46.67% | 3.54 pp | -54 | 45 | -1.20 |
| BTC Daily | lstm | LSTM | 762 | 322 | 440 | 42.26% | 35.83% | 40.62% | 7.74 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 762 | 319 | 443 | 41.86% | 38.33% | 42.29% | 8.14 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 772 | 304 | 468 | 39.38% | 35.42% | 37.29% | 10.62 pp | -164 | 45 | -3.64 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 534 | 259 | 275 | 48.50% | 45.42% | 48.54% | 1.50 pp | -16 | 50 | -0.32 |
| BTC Market Hours | transformer | Transformer | 534 | 256 | 278 | 47.94% | 48.75% | 48.54% | 2.06 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 534 | 253 | 281 | 47.38% | 50.42% | 48.96% | 2.62 pp | -28 | 50 | -0.56 |
| BTC Market Hours | lstm | LSTM | 534 | 231 | 303 | 43.26% | 42.08% | 44.17% | 6.74 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 534 | 230 | 304 | 43.07% | 44.17% | 43.75% | 6.93 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 534 | 222 | 312 | 41.57% | 42.92% | 42.29% | 8.43 pp | -90 | 50 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 588 | 279 | 309 | 47.45% | 50.83% | 48.54% | 2.55 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 588 | 275 | 313 | 46.77% | 46.67% | 48.12% | 3.23 pp | -38 | 50 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 588 | 274 | 314 | 46.60% | 51.25% | 47.50% | 3.40 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 588 | 245 | 343 | 41.67% | 44.58% | 41.46% | 8.33 pp | -98 | 50 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 588 | 239 | 349 | 40.65% | 40.00% | 40.42% | 9.35 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 588 | 235 | 353 | 39.97% | 41.67% | 38.96% | 10.03 pp | -118 | 50 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
