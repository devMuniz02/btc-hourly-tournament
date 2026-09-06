# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T14:22:47.466283+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 180 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 534 | 259 | 275 | 48.50% | 45.42% | 48.54% | 1.50 pp | -16 | 50 | -0.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| BTC Market Hours | transformer | Transformer | 534 | 256 | 278 | 47.94% | 48.75% | 48.54% | 2.06 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 762 | 370 | 392 | 48.56% | 47.92% | 48.54% | 1.44 pp | -22 | 45 | -0.49 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| BTC Market Hours | nn | NN | 534 | 253 | 281 | 47.38% | 50.42% | 48.96% | 2.62 pp | -28 | 50 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 588 | 279 | 309 | 47.45% | 50.83% | 48.54% | 2.55 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 588 | 275 | 313 | 46.77% | 46.67% | 48.12% | 3.23 pp | -38 | 50 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 588 | 274 | 314 | 46.60% | 51.25% | 47.50% | 3.40 pp | -40 | 50 | -0.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 5 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 939 | 448 | 491 | 47.71% | 50.00% | 46.88% | 2.29 pp | -43 | 49 | -0.88 |
| BTC Daily | transformer | Transformer | 762 | 359 | 403 | 47.11% | 42.92% | 47.92% | 2.89 pp | -44 | 45 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| BTC Hourly | transformer | Transformer | 939 | 441 | 498 | 46.96% | 46.67% | 45.42% | 3.04 pp | -57 | 49 | -1.16 |
| BTC Daily | nn | NN | 762 | 354 | 408 | 46.46% | 45.42% | 46.67% | 3.54 pp | -54 | 45 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| BTC Market Hours | lstm | LSTM | 534 | 231 | 303 | 43.26% | 42.08% | 44.17% | 6.74 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 534 | 230 | 304 | 43.07% | 44.17% | 43.75% | 6.93 pp | -74 | 50 | -1.48 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 534 | 222 | 312 | 41.57% | 42.92% | 42.29% | 8.43 pp | -90 | 50 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 588 | 245 | 343 | 41.67% | 44.58% | 41.46% | 8.33 pp | -98 | 50 | -1.96 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 939 | 418 | 521 | 44.52% | 44.58% | 43.75% | 5.48 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 939 | 417 | 522 | 44.41% | 42.50% | 42.29% | 5.59 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 588 | 239 | 349 | 40.65% | 40.00% | 40.42% | 9.35 pp | -110 | 50 | -2.20 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 588 | 235 | 353 | 39.97% | 41.67% | 38.96% | 10.03 pp | -118 | 50 | -2.36 |
| BTC Daily | lstm | LSTM | 762 | 322 | 440 | 42.26% | 35.83% | 40.62% | 7.74 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 762 | 319 | 443 | 41.86% | 38.33% | 42.29% | 8.14 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 939 | 400 | 539 | 42.60% | 36.25% | 41.67% | 7.40 pp | -139 | 49 | -2.84 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 939 | 395 | 544 | 42.07% | 41.25% | 40.62% | 7.93 pp | -149 | 49 | -3.04 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
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
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
