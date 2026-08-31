# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T20:45:40.564663+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1168 | 880 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1044 | 679 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 19:00:00+00:00 | 668 | 441 | 226 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 19:00:00+00:00 | 669 | 494 | 173 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 93 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 93 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 6 | 87 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 16:00:00+00:00 | 93 | 6 | 87 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 441 | 215 | 226 | 48.75% | 44.58% | 48.75% | 1.25 pp | -11 | 43 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 669 | 327 | 342 | 48.88% | 47.08% | 49.79% | 1.12 pp | -15 | 41 | -0.37 |
| BTC Daily | transformer | Transformer | 669 | 324 | 345 | 48.43% | 46.25% | 49.79% | 1.57 pp | -21 | 41 | -0.51 |
| Consolidated Hourly | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 441 | 208 | 233 | 47.17% | 48.75% | 47.17% | 2.83 pp | -25 | 43 | -0.58 |
| Consolidated Hourly | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 441 | 202 | 239 | 45.80% | 40.83% | 45.80% | 4.20 pp | -37 | 43 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 494 | 228 | 266 | 46.15% | 47.08% | 46.46% | 3.85 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | nn | NN | 494 | 226 | 268 | 45.75% | 43.75% | 46.25% | 4.25 pp | -42 | 43 | -0.98 |
| BTC Daily | nn | NN | 669 | 314 | 355 | 46.94% | 43.75% | 49.58% | 3.06 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |
| BTC Hourly | transformer | Transformer | 846 | 400 | 446 | 47.28% | 47.92% | 47.29% | 2.72 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 494 | 224 | 270 | 45.34% | 45.00% | 45.21% | 4.66 pp | -46 | 43 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 846 | 397 | 449 | 46.93% | 44.17% | 46.88% | 3.07 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 441 | 189 | 252 | 42.86% | 41.67% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 441 | 189 | 252 | 42.86% | 43.33% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Hourly | nn | NN | 846 | 381 | 465 | 45.04% | 43.75% | 44.58% | 4.96 pp | -84 | 45 | -1.87 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 441 | 177 | 264 | 40.14% | 38.75% | 40.14% | 9.86 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 669 | 293 | 376 | 43.80% | 39.17% | 43.12% | 6.20 pp | -83 | 41 | -2.02 |
| BTC Hourly | rf | RandomForest | 846 | 377 | 469 | 44.56% | 43.75% | 44.17% | 5.44 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 494 | 203 | 291 | 41.09% | 41.25% | 41.25% | 8.91 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 494 | 197 | 297 | 39.88% | 37.50% | 40.62% | 10.12 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 669 | 286 | 383 | 42.75% | 40.42% | 43.96% | 7.25 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 494 | 193 | 301 | 39.07% | 36.25% | 39.17% | 10.93 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 846 | 363 | 483 | 42.91% | 40.00% | 42.50% | 7.09 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 846 | 356 | 490 | 42.08% | 40.00% | 42.50% | 7.92 pp | -134 | 45 | -2.98 |
| BTC Daily | xgb | XGBoost | 679 | 270 | 409 | 39.76% | 34.58% | 39.79% | 10.24 pp | -139 | 41 | -3.39 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 846 | 400 | 446 | 47.28% | 47.92% | 47.29% | 2.72 pp | -46 | 45 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 846 | 397 | 449 | 46.93% | 44.17% | 46.88% | 3.07 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 846 | 381 | 465 | 45.04% | 43.75% | 44.58% | 4.96 pp | -84 | 45 | -1.87 |
| BTC Hourly | rf | RandomForest | 846 | 377 | 469 | 44.56% | 43.75% | 44.17% | 5.44 pp | -92 | 45 | -2.04 |
| BTC Hourly | lstm | LSTM | 846 | 363 | 483 | 42.91% | 40.00% | 42.50% | 7.09 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 846 | 356 | 490 | 42.08% | 40.00% | 42.50% | 7.92 pp | -134 | 45 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 669 | 327 | 342 | 48.88% | 47.08% | 49.79% | 1.12 pp | -15 | 41 | -0.37 |
| BTC Daily | transformer | Transformer | 669 | 324 | 345 | 48.43% | 46.25% | 49.79% | 1.57 pp | -21 | 41 | -0.51 |
| BTC Daily | nn | NN | 669 | 314 | 355 | 46.94% | 43.75% | 49.58% | 3.06 pp | -41 | 41 | -1.00 |
| BTC Daily | lstm | LSTM | 669 | 293 | 376 | 43.80% | 39.17% | 43.12% | 6.20 pp | -83 | 41 | -2.02 |
| BTC Daily | rf | RandomForest | 669 | 286 | 383 | 42.75% | 40.42% | 43.96% | 7.25 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 679 | 270 | 409 | 39.76% | 34.58% | 39.79% | 10.24 pp | -139 | 41 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 441 | 215 | 226 | 48.75% | 44.58% | 48.75% | 1.25 pp | -11 | 43 | -0.26 |
| BTC Market Hours | nn | NN | 441 | 208 | 233 | 47.17% | 48.75% | 47.17% | 2.83 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 441 | 202 | 239 | 45.80% | 40.83% | 45.80% | 4.20 pp | -37 | 43 | -0.86 |
| BTC Market Hours | lstm | LSTM | 441 | 189 | 252 | 42.86% | 41.67% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 441 | 189 | 252 | 42.86% | 43.33% | 42.86% | 7.14 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 441 | 177 | 264 | 40.14% | 38.75% | 40.14% | 9.86 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 494 | 228 | 266 | 46.15% | 47.08% | 46.46% | 3.85 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | nn | NN | 494 | 226 | 268 | 45.75% | 43.75% | 46.25% | 4.25 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 494 | 224 | 270 | 45.34% | 45.00% | 45.21% | 4.66 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 494 | 203 | 291 | 41.09% | 41.25% | 41.25% | 8.91 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 494 | 197 | 297 | 39.88% | 37.50% | 40.62% | 10.12 pp | -100 | 43 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 494 | 193 | 301 | 39.07% | 36.25% | 39.17% | 10.93 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 50 | 43 | 53.76% | 53.76% | 53.76% | 3.76 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
