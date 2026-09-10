# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T06:10:43.389234+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1320 | 1032 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1195 | 830 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 941 | 592 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 943 | 646 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 592 | 287 | 305 | 48.48% | 47.50% | 47.29% | 1.52 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 592 | 286 | 306 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| BTC Market Hours | transformer | Transformer | 592 | 278 | 314 | 46.96% | 46.67% | 46.25% | 3.04 pp | -36 | 55 | -0.65 |
| BTC Daily | mlp_sklearn | MLPClassifier | 820 | 393 | 427 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 646 | 303 | 343 | 46.90% | 48.75% | 47.29% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 646 | 303 | 343 | 46.90% | 48.33% | 48.12% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 646 | 301 | 345 | 46.59% | 48.33% | 47.71% | 3.41 pp | -44 | 55 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 998 | 473 | 525 | 47.39% | 49.17% | 46.04% | 2.61 pp | -52 | 52 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| BTC Daily | nn | NN | 820 | 382 | 438 | 46.59% | 45.42% | 45.42% | 3.41 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| BTC Daily | transformer | Transformer | 820 | 379 | 441 | 46.22% | 38.75% | 45.42% | 3.78 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 998 | 464 | 534 | 46.49% | 45.42% | 44.79% | 3.51 pp | -70 | 52 | -1.35 |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 592 | 254 | 338 | 42.91% | 45.00% | 43.54% | 7.09 pp | -84 | 55 | -1.53 |
| BTC Market Hours | lstm | LSTM | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 646 | 267 | 379 | 41.33% | 41.67% | 40.62% | 8.67 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 646 | 266 | 380 | 41.18% | 42.92% | 40.42% | 8.82 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 646 | 263 | 383 | 40.71% | 41.67% | 40.00% | 9.29 pp | -120 | 55 | -2.18 |
| BTC Hourly | nn | NN | 998 | 440 | 558 | 44.09% | 42.50% | 41.67% | 5.91 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 998 | 439 | 559 | 43.99% | 41.25% | 43.12% | 6.01 pp | -120 | 52 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| BTC Daily | lstm | LSTM | 820 | 348 | 472 | 42.44% | 36.25% | 41.04% | 7.56 pp | -124 | 47 | -2.64 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 998 | 423 | 575 | 42.38% | 37.50% | 40.00% | 7.62 pp | -152 | 52 | -2.92 |
| BTC Daily | rf | RandomForest | 820 | 341 | 479 | 41.59% | 37.92% | 41.04% | 8.41 pp | -138 | 47 | -2.94 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| BTC Hourly | xgb | XGBoost | 998 | 410 | 588 | 41.08% | 35.00% | 38.54% | 8.92 pp | -178 | 52 | -3.42 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 830 | 327 | 503 | 39.40% | 37.92% | 36.25% | 10.60 pp | -176 | 47 | -3.74 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 998 | 473 | 525 | 47.39% | 49.17% | 46.04% | 2.61 pp | -52 | 52 | -1.00 |
| BTC Hourly | transformer | Transformer | 998 | 464 | 534 | 46.49% | 45.42% | 44.79% | 3.51 pp | -70 | 52 | -1.35 |
| BTC Hourly | nn | NN | 998 | 440 | 558 | 44.09% | 42.50% | 41.67% | 5.91 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 998 | 439 | 559 | 43.99% | 41.25% | 43.12% | 6.01 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 998 | 423 | 575 | 42.38% | 37.50% | 40.00% | 7.62 pp | -152 | 52 | -2.92 |
| BTC Hourly | xgb | XGBoost | 998 | 410 | 588 | 41.08% | 35.00% | 38.54% | 8.92 pp | -178 | 52 | -3.42 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 820 | 393 | 427 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 820 | 382 | 438 | 46.59% | 45.42% | 45.42% | 3.41 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 820 | 379 | 441 | 46.22% | 38.75% | 45.42% | 3.78 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 820 | 348 | 472 | 42.44% | 36.25% | 41.04% | 7.56 pp | -124 | 47 | -2.64 |
| BTC Daily | rf | RandomForest | 820 | 341 | 479 | 41.59% | 37.92% | 41.04% | 8.41 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 830 | 327 | 503 | 39.40% | 37.92% | 36.25% | 10.60 pp | -176 | 47 | -3.74 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 592 | 287 | 305 | 48.48% | 47.50% | 47.29% | 1.52 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 592 | 286 | 306 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 592 | 278 | 314 | 46.96% | 46.67% | 46.25% | 3.04 pp | -36 | 55 | -0.65 |
| BTC Market Hours | xgb | XGBoost | 592 | 254 | 338 | 42.91% | 45.00% | 43.54% | 7.09 pp | -84 | 55 | -1.53 |
| BTC Market Hours | lstm | LSTM | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 646 | 303 | 343 | 46.90% | 48.75% | 47.29% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 646 | 303 | 343 | 46.90% | 48.33% | 48.12% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 646 | 301 | 345 | 46.59% | 48.33% | 47.71% | 3.41 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 646 | 267 | 379 | 41.33% | 41.67% | 40.62% | 8.67 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 646 | 266 | 380 | 41.18% | 42.92% | 40.42% | 8.82 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 646 | 263 | 383 | 40.71% | 41.67% | 40.00% | 9.29 pp | -120 | 55 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
