# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T12:03:48.937686+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1323 | 1035 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1199 | 834 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 945 | 596 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 947 | 650 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 596 | 287 | 309 | 48.15% | 46.25% | 47.08% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 596 | 287 | 309 | 48.15% | 51.67% | 49.79% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | transformer | Transformer | 596 | 280 | 316 | 46.98% | 46.67% | 46.46% | 3.02 pp | -36 | 55 | -0.65 |
| BTC Daily | mlp_sklearn | MLPClassifier | 824 | 395 | 429 | 47.94% | 45.00% | 46.88% | 2.06 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | nn | NN | 650 | 305 | 345 | 46.92% | 48.33% | 47.92% | 3.08 pp | -40 | 55 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 650 | 304 | 346 | 46.77% | 48.33% | 47.29% | 3.23 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 650 | 303 | 347 | 46.62% | 47.92% | 47.92% | 3.38 pp | -44 | 55 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1001 | 475 | 526 | 47.45% | 48.75% | 46.25% | 2.55 pp | -51 | 52 | -0.98 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 824 | 384 | 440 | 46.60% | 45.42% | 45.42% | 3.40 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Daily | transformer | Transformer | 824 | 381 | 443 | 46.24% | 37.92% | 45.00% | 3.76 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1001 | 466 | 535 | 46.55% | 45.83% | 44.79% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Market Hours | lstm | LSTM | 596 | 255 | 341 | 42.79% | 42.08% | 42.50% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 596 | 255 | 341 | 42.79% | 45.00% | 43.54% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 596 | 254 | 342 | 42.62% | 42.50% | 42.08% | 7.38 pp | -88 | 55 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 650 | 269 | 381 | 41.38% | 41.67% | 41.04% | 8.62 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 650 | 268 | 382 | 41.23% | 42.92% | 40.62% | 8.77 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 650 | 265 | 385 | 40.77% | 42.08% | 40.21% | 9.23 pp | -120 | 55 | -2.18 |
| BTC Hourly | nn | NN | 1001 | 443 | 558 | 44.26% | 42.50% | 41.88% | 5.74 pp | -115 | 52 | -2.21 |
| BTC Hourly | rf | RandomForest | 1001 | 442 | 559 | 44.16% | 42.08% | 43.54% | 5.84 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 824 | 348 | 476 | 42.23% | 35.00% | 40.62% | 7.77 pp | -128 | 47 | -2.72 |
| BTC Hourly | lstm | LSTM | 1001 | 425 | 576 | 42.46% | 37.08% | 40.21% | 7.54 pp | -151 | 52 | -2.90 |
| BTC Daily | rf | RandomForest | 824 | 343 | 481 | 41.63% | 37.50% | 41.04% | 8.37 pp | -138 | 47 | -2.94 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| BTC Hourly | xgb | XGBoost | 1001 | 413 | 588 | 41.26% | 35.83% | 38.96% | 8.74 pp | -175 | 52 | -3.37 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 834 | 329 | 505 | 39.45% | 37.08% | 36.67% | 10.55 pp | -176 | 47 | -3.74 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1001 | 475 | 526 | 47.45% | 48.75% | 46.25% | 2.55 pp | -51 | 52 | -0.98 |
| BTC Hourly | transformer | Transformer | 1001 | 466 | 535 | 46.55% | 45.83% | 44.79% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1001 | 443 | 558 | 44.26% | 42.50% | 41.88% | 5.74 pp | -115 | 52 | -2.21 |
| BTC Hourly | rf | RandomForest | 1001 | 442 | 559 | 44.16% | 42.08% | 43.54% | 5.84 pp | -117 | 52 | -2.25 |
| BTC Hourly | lstm | LSTM | 1001 | 425 | 576 | 42.46% | 37.08% | 40.21% | 7.54 pp | -151 | 52 | -2.90 |
| BTC Hourly | xgb | XGBoost | 1001 | 413 | 588 | 41.26% | 35.83% | 38.96% | 8.74 pp | -175 | 52 | -3.37 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 824 | 395 | 429 | 47.94% | 45.00% | 46.88% | 2.06 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 824 | 384 | 440 | 46.60% | 45.42% | 45.42% | 3.40 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 824 | 381 | 443 | 46.24% | 37.92% | 45.00% | 3.76 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 824 | 348 | 476 | 42.23% | 35.00% | 40.62% | 7.77 pp | -128 | 47 | -2.72 |
| BTC Daily | rf | RandomForest | 824 | 343 | 481 | 41.63% | 37.50% | 41.04% | 8.37 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 834 | 329 | 505 | 39.45% | 37.08% | 36.67% | 10.55 pp | -176 | 47 | -3.74 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 596 | 287 | 309 | 48.15% | 46.25% | 47.08% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 596 | 287 | 309 | 48.15% | 51.67% | 49.79% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | transformer | Transformer | 596 | 280 | 316 | 46.98% | 46.67% | 46.46% | 3.02 pp | -36 | 55 | -0.65 |
| BTC Market Hours | lstm | LSTM | 596 | 255 | 341 | 42.79% | 42.08% | 42.50% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 596 | 255 | 341 | 42.79% | 45.00% | 43.54% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 596 | 254 | 342 | 42.62% | 42.50% | 42.08% | 7.38 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 650 | 305 | 345 | 46.92% | 48.33% | 47.92% | 3.08 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 650 | 304 | 346 | 46.77% | 48.33% | 47.29% | 3.23 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 650 | 303 | 347 | 46.62% | 47.92% | 47.92% | 3.38 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 650 | 269 | 381 | 41.38% | 41.67% | 41.04% | 8.62 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 650 | 268 | 382 | 41.23% | 42.92% | 40.62% | 8.77 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 650 | 265 | 385 | 40.77% | 42.08% | 40.21% | 9.23 pp | -120 | 55 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
