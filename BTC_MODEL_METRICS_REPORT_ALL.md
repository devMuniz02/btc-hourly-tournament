# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T13:35:24.532379+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1260 | 972 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1136 | 771 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 12:00:00+00:00 | 831 | 533 | 297 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 12:00:00+00:00 | 833 | 587 | 244 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 533 | 259 | 274 | 48.59% | 45.42% | 48.54% | 1.41 pp | -15 | 50 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 533 | 255 | 278 | 47.84% | 48.33% | 48.54% | 2.16 pp | -23 | 50 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 761 | 370 | 391 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 45 | -0.47 |
| BTC Market Hours | nn | NN | 533 | 253 | 280 | 47.47% | 50.83% | 48.96% | 2.53 pp | -27 | 50 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 587 | 279 | 308 | 47.53% | 51.25% | 48.75% | 2.47 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 587 | 274 | 313 | 46.68% | 46.67% | 47.92% | 3.32 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 587 | 273 | 314 | 46.51% | 51.25% | 47.29% | 3.49 pp | -41 | 50 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 938 | 448 | 490 | 47.76% | 50.00% | 47.08% | 2.24 pp | -42 | 49 | -0.86 |
| BTC Daily | transformer | Transformer | 761 | 358 | 403 | 47.04% | 42.92% | 47.92% | 2.96 pp | -45 | 45 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 938 | 441 | 497 | 47.01% | 46.67% | 45.42% | 2.99 pp | -56 | 49 | -1.14 |
| BTC Daily | nn | NN | 761 | 353 | 408 | 46.39% | 45.00% | 46.46% | 3.61 pp | -55 | 45 | -1.22 |
| BTC Market Hours | lstm | LSTM | 533 | 231 | 302 | 43.34% | 42.50% | 44.17% | 6.66 pp | -71 | 50 | -1.42 |
| BTC Market Hours | rf | RandomForest | 533 | 230 | 303 | 43.15% | 44.17% | 43.75% | 6.85 pp | -73 | 50 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 533 | 222 | 311 | 41.65% | 42.92% | 42.29% | 8.35 pp | -89 | 50 | -1.78 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 587 | 244 | 343 | 41.57% | 44.58% | 41.25% | 8.43 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 938 | 418 | 520 | 44.56% | 44.58% | 43.96% | 5.44 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 938 | 417 | 521 | 44.46% | 42.50% | 42.50% | 5.54 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 587 | 238 | 349 | 40.55% | 40.00% | 40.42% | 9.45 pp | -111 | 50 | -2.22 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 587 | 234 | 353 | 39.86% | 41.67% | 38.96% | 10.14 pp | -119 | 50 | -2.38 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 761 | 322 | 439 | 42.31% | 35.83% | 40.62% | 7.69 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 761 | 318 | 443 | 41.79% | 37.92% | 42.08% | 8.21 pp | -125 | 45 | -2.78 |
| BTC Hourly | lstm | LSTM | 938 | 400 | 538 | 42.64% | 36.67% | 41.67% | 7.36 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 938 | 395 | 543 | 42.11% | 41.67% | 40.62% | 7.89 pp | -148 | 49 | -3.02 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 771 | 303 | 468 | 39.30% | 35.42% | 37.08% | 10.70 pp | -165 | 45 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 938 | 448 | 490 | 47.76% | 50.00% | 47.08% | 2.24 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 938 | 441 | 497 | 47.01% | 46.67% | 45.42% | 2.99 pp | -56 | 49 | -1.14 |
| BTC Hourly | rf | RandomForest | 938 | 418 | 520 | 44.56% | 44.58% | 43.96% | 5.44 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 938 | 417 | 521 | 44.46% | 42.50% | 42.50% | 5.54 pp | -104 | 49 | -2.12 |
| BTC Hourly | lstm | LSTM | 938 | 400 | 538 | 42.64% | 36.67% | 41.67% | 7.36 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 938 | 395 | 543 | 42.11% | 41.67% | 40.62% | 7.89 pp | -148 | 49 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 761 | 370 | 391 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 45 | -0.47 |
| BTC Daily | transformer | Transformer | 761 | 358 | 403 | 47.04% | 42.92% | 47.92% | 2.96 pp | -45 | 45 | -1.00 |
| BTC Daily | nn | NN | 761 | 353 | 408 | 46.39% | 45.00% | 46.46% | 3.61 pp | -55 | 45 | -1.22 |
| BTC Daily | lstm | LSTM | 761 | 322 | 439 | 42.31% | 35.83% | 40.62% | 7.69 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 761 | 318 | 443 | 41.79% | 37.92% | 42.08% | 8.21 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 771 | 303 | 468 | 39.30% | 35.42% | 37.08% | 10.70 pp | -165 | 45 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 533 | 259 | 274 | 48.59% | 45.42% | 48.54% | 1.41 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 533 | 255 | 278 | 47.84% | 48.33% | 48.54% | 2.16 pp | -23 | 50 | -0.46 |
| BTC Market Hours | nn | NN | 533 | 253 | 280 | 47.47% | 50.83% | 48.96% | 2.53 pp | -27 | 50 | -0.54 |
| BTC Market Hours | lstm | LSTM | 533 | 231 | 302 | 43.34% | 42.50% | 44.17% | 6.66 pp | -71 | 50 | -1.42 |
| BTC Market Hours | rf | RandomForest | 533 | 230 | 303 | 43.15% | 44.17% | 43.75% | 6.85 pp | -73 | 50 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 533 | 222 | 311 | 41.65% | 42.92% | 42.29% | 8.35 pp | -89 | 50 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 587 | 279 | 308 | 47.53% | 51.25% | 48.75% | 2.47 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 587 | 274 | 313 | 46.68% | 46.67% | 47.92% | 3.32 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 587 | 273 | 314 | 46.51% | 51.25% | 47.29% | 3.49 pp | -41 | 50 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 587 | 244 | 343 | 41.57% | 44.58% | 41.25% | 8.43 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 587 | 238 | 349 | 40.55% | 40.00% | 40.42% | 9.45 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 587 | 234 | 353 | 39.86% | 41.67% | 38.96% | 10.14 pp | -119 | 50 | -2.38 |

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
