# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T17:44:58.405698+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1166 | 878 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1041 | 676 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 16:00:00+00:00 | 662 | 438 | 223 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 16:00:00+00:00 | 664 | 492 | 170 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 438 | 215 | 223 | 49.09% | 45.00% | 49.09% | 0.91 pp | -8 | 43 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 666 | 325 | 341 | 48.80% | 46.67% | 49.79% | 1.20 pp | -16 | 41 | -0.39 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| BTC Daily | transformer | Transformer | 666 | 321 | 345 | 48.20% | 45.42% | 49.58% | 1.80 pp | -24 | 41 | -0.59 |
| BTC Market Hours | nn | NN | 438 | 206 | 232 | 47.03% | 48.75% | 47.03% | 2.97 pp | -26 | 43 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 438 | 202 | 236 | 46.12% | 41.67% | 46.12% | 3.88 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 492 | 228 | 264 | 46.34% | 47.08% | 46.46% | 3.66 pp | -36 | 43 | -0.84 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 844 | 399 | 445 | 47.27% | 47.92% | 47.08% | 2.73 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | nn | NN | 492 | 224 | 268 | 45.53% | 43.33% | 46.04% | 4.47 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 492 | 224 | 268 | 45.53% | 45.42% | 45.42% | 4.47 pp | -44 | 43 | -1.02 |
| BTC Daily | nn | NN | 666 | 312 | 354 | 46.85% | 42.92% | 49.38% | 3.15 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 844 | 396 | 448 | 46.92% | 44.17% | 46.67% | 3.08 pp | -52 | 45 | -1.16 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| BTC Market Hours | lstm | LSTM | 438 | 188 | 250 | 42.92% | 41.67% | 42.92% | 7.08 pp | -62 | 43 | -1.44 |
| BTC Market Hours | rf | RandomForest | 438 | 188 | 250 | 42.92% | 43.33% | 42.92% | 7.08 pp | -62 | 43 | -1.44 |
| BTC Hourly | nn | NN | 844 | 381 | 463 | 45.14% | 44.17% | 44.58% | 4.86 pp | -82 | 45 | -1.82 |
| BTC Daily | lstm | LSTM | 666 | 292 | 374 | 43.84% | 39.17% | 43.33% | 6.16 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 492 | 203 | 289 | 41.26% | 41.25% | 41.46% | 8.74 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 438 | 176 | 262 | 40.18% | 38.75% | 40.18% | 9.82 pp | -86 | 43 | -2.00 |
| BTC Hourly | rf | RandomForest | 844 | 376 | 468 | 44.55% | 43.75% | 43.96% | 5.45 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 492 | 197 | 295 | 40.04% | 38.33% | 40.62% | 9.96 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 666 | 284 | 382 | 42.64% | 40.42% | 43.54% | 7.36 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 492 | 193 | 299 | 39.23% | 36.25% | 39.38% | 10.77 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 844 | 362 | 482 | 42.89% | 40.00% | 42.29% | 7.11 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 844 | 356 | 488 | 42.18% | 40.00% | 42.50% | 7.82 pp | -132 | 45 | -2.93 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 676 | 268 | 408 | 39.64% | 34.17% | 39.58% | 10.36 pp | -140 | 41 | -3.41 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 844 | 399 | 445 | 47.27% | 47.92% | 47.08% | 2.73 pp | -46 | 45 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 844 | 396 | 448 | 46.92% | 44.17% | 46.67% | 3.08 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 844 | 381 | 463 | 45.14% | 44.17% | 44.58% | 4.86 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 844 | 376 | 468 | 44.55% | 43.75% | 43.96% | 5.45 pp | -92 | 45 | -2.04 |
| BTC Hourly | lstm | LSTM | 844 | 362 | 482 | 42.89% | 40.00% | 42.29% | 7.11 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 844 | 356 | 488 | 42.18% | 40.00% | 42.50% | 7.82 pp | -132 | 45 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 666 | 325 | 341 | 48.80% | 46.67% | 49.79% | 1.20 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 666 | 321 | 345 | 48.20% | 45.42% | 49.58% | 1.80 pp | -24 | 41 | -0.59 |
| BTC Daily | nn | NN | 666 | 312 | 354 | 46.85% | 42.92% | 49.38% | 3.15 pp | -42 | 41 | -1.02 |
| BTC Daily | lstm | LSTM | 666 | 292 | 374 | 43.84% | 39.17% | 43.33% | 6.16 pp | -82 | 41 | -2.00 |
| BTC Daily | rf | RandomForest | 666 | 284 | 382 | 42.64% | 40.42% | 43.54% | 7.36 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 676 | 268 | 408 | 39.64% | 34.17% | 39.58% | 10.36 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 438 | 215 | 223 | 49.09% | 45.00% | 49.09% | 0.91 pp | -8 | 43 | -0.19 |
| BTC Market Hours | nn | NN | 438 | 206 | 232 | 47.03% | 48.75% | 47.03% | 2.97 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 438 | 202 | 236 | 46.12% | 41.67% | 46.12% | 3.88 pp | -34 | 43 | -0.79 |
| BTC Market Hours | lstm | LSTM | 438 | 188 | 250 | 42.92% | 41.67% | 42.92% | 7.08 pp | -62 | 43 | -1.44 |
| BTC Market Hours | rf | RandomForest | 438 | 188 | 250 | 42.92% | 43.33% | 42.92% | 7.08 pp | -62 | 43 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 438 | 176 | 262 | 40.18% | 38.75% | 40.18% | 9.82 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 492 | 228 | 264 | 46.34% | 47.08% | 46.46% | 3.66 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | nn | NN | 492 | 224 | 268 | 45.53% | 43.33% | 46.04% | 4.47 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 492 | 224 | 268 | 45.53% | 45.42% | 45.42% | 4.47 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 492 | 203 | 289 | 41.26% | 41.25% | 41.46% | 8.74 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 492 | 197 | 295 | 40.04% | 38.33% | 40.62% | 9.96 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 492 | 193 | 299 | 39.23% | 36.25% | 39.38% | 10.77 pp | -106 | 43 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
