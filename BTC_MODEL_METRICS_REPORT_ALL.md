# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T22:12:37.898142+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1201 | 913 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1077 | 712 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 21:00:00+00:00 | 729 | 474 | 254 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 21:00:00+00:00 | 731 | 528 | 201 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 126 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 474 | 229 | 245 | 48.31% | 43.33% | 48.31% | 1.69 pp | -16 | 46 | -0.35 |
| BTC Daily | mlp_sklearn | MLPClassifier | 702 | 343 | 359 | 48.86% | 47.08% | 48.96% | 1.14 pp | -16 | 42 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 474 | 223 | 251 | 47.05% | 47.50% | 47.05% | 2.95 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 702 | 338 | 364 | 48.15% | 47.50% | 50.00% | 1.85 pp | -26 | 42 | -0.62 |
| BTC Market Hours | transformer | Transformer | 474 | 219 | 255 | 46.20% | 40.42% | 46.20% | 3.80 pp | -36 | 46 | -0.78 |
| Consolidated Hourly | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 528 | 243 | 285 | 46.02% | 47.92% | 46.67% | 3.98 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | nn | NN | 528 | 242 | 286 | 45.83% | 43.33% | 46.46% | 4.17 pp | -44 | 46 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 879 | 416 | 463 | 47.33% | 48.33% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | transformer | Transformer | 879 | 416 | 463 | 47.33% | 48.75% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 528 | 241 | 287 | 45.64% | 46.67% | 46.25% | 4.36 pp | -46 | 46 | -1.00 |
| BTC Daily | nn | NN | 702 | 326 | 376 | 46.44% | 43.33% | 48.54% | 3.56 pp | -50 | 42 | -1.19 |
| BTC Market Hours | rf | RandomForest | 474 | 204 | 270 | 43.04% | 42.50% | 43.04% | 6.96 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 474 | 203 | 271 | 42.83% | 40.42% | 42.83% | 7.17 pp | -68 | 46 | -1.48 |
| Consolidated Hourly | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| BTC Hourly | nn | NN | 879 | 395 | 484 | 44.94% | 45.83% | 43.96% | 5.06 pp | -89 | 47 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 474 | 193 | 281 | 40.72% | 39.58% | 40.72% | 9.28 pp | -88 | 46 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 528 | 219 | 309 | 41.48% | 42.08% | 41.67% | 8.52 pp | -90 | 46 | -1.96 |
| BTC Hourly | rf | RandomForest | 879 | 391 | 488 | 44.48% | 44.58% | 44.38% | 5.52 pp | -97 | 47 | -2.06 |
| BTC Daily | lstm | LSTM | 702 | 305 | 397 | 43.45% | 38.75% | 42.29% | 6.55 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 702 | 302 | 400 | 43.02% | 41.67% | 43.54% | 6.98 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 528 | 210 | 318 | 39.77% | 36.67% | 40.62% | 10.23 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 528 | 210 | 318 | 39.77% | 37.92% | 39.17% | 10.23 pp | -108 | 46 | -2.35 |
| BTC Hourly | lstm | LSTM | 879 | 375 | 504 | 42.66% | 38.33% | 42.08% | 7.34 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 879 | 372 | 507 | 42.32% | 41.67% | 42.92% | 7.68 pp | -135 | 47 | -2.87 |
| BTC Daily | xgb | XGBoost | 712 | 282 | 430 | 39.61% | 35.42% | 39.38% | 10.39 pp | -148 | 42 | -3.52 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 879 | 416 | 463 | 47.33% | 48.33% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | transformer | Transformer | 879 | 416 | 463 | 47.33% | 48.75% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 879 | 395 | 484 | 44.94% | 45.83% | 43.96% | 5.06 pp | -89 | 47 | -1.89 |
| BTC Hourly | rf | RandomForest | 879 | 391 | 488 | 44.48% | 44.58% | 44.38% | 5.52 pp | -97 | 47 | -2.06 |
| BTC Hourly | lstm | LSTM | 879 | 375 | 504 | 42.66% | 38.33% | 42.08% | 7.34 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 879 | 372 | 507 | 42.32% | 41.67% | 42.92% | 7.68 pp | -135 | 47 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 702 | 343 | 359 | 48.86% | 47.08% | 48.96% | 1.14 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 702 | 338 | 364 | 48.15% | 47.50% | 50.00% | 1.85 pp | -26 | 42 | -0.62 |
| BTC Daily | nn | NN | 702 | 326 | 376 | 46.44% | 43.33% | 48.54% | 3.56 pp | -50 | 42 | -1.19 |
| BTC Daily | lstm | LSTM | 702 | 305 | 397 | 43.45% | 38.75% | 42.29% | 6.55 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 702 | 302 | 400 | 43.02% | 41.67% | 43.54% | 6.98 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 712 | 282 | 430 | 39.61% | 35.42% | 39.38% | 10.39 pp | -148 | 42 | -3.52 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 474 | 229 | 245 | 48.31% | 43.33% | 48.31% | 1.69 pp | -16 | 46 | -0.35 |
| BTC Market Hours | nn | NN | 474 | 223 | 251 | 47.05% | 47.50% | 47.05% | 2.95 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 474 | 219 | 255 | 46.20% | 40.42% | 46.20% | 3.80 pp | -36 | 46 | -0.78 |
| BTC Market Hours | rf | RandomForest | 474 | 204 | 270 | 43.04% | 42.50% | 43.04% | 6.96 pp | -66 | 46 | -1.43 |
| BTC Market Hours | lstm | LSTM | 474 | 203 | 271 | 42.83% | 40.42% | 42.83% | 7.17 pp | -68 | 46 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 474 | 193 | 281 | 40.72% | 39.58% | 40.72% | 9.28 pp | -88 | 46 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 528 | 243 | 285 | 46.02% | 47.92% | 46.67% | 3.98 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | nn | NN | 528 | 242 | 286 | 45.83% | 43.33% | 46.46% | 4.17 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 528 | 241 | 287 | 45.64% | 46.67% | 46.25% | 4.36 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 528 | 219 | 309 | 41.48% | 42.08% | 41.67% | 8.52 pp | -90 | 46 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 528 | 210 | 318 | 39.77% | 36.67% | 40.62% | 10.23 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 528 | 210 | 318 | 39.77% | 37.92% | 39.17% | 10.23 pp | -108 | 46 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
