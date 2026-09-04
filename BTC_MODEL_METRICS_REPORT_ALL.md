# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T21:21:29.478026+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1233 | 945 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1109 | 744 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 786 | 506 | 279 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 788 | 560 | 226 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 154 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T23:00:00+00:00 | 155 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 506 | 244 | 262 | 48.22% | 44.58% | 48.12% | 1.78 pp | -18 | 48 | -0.38 |
| BTC Market Hours | transformer | Transformer | 506 | 242 | 264 | 47.83% | 46.25% | 48.33% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 734 | 356 | 378 | 48.50% | 47.50% | 48.54% | 1.50 pp | -22 | 43 | -0.51 |
| BTC Market Hours | nn | NN | 506 | 239 | 267 | 47.23% | 50.00% | 48.12% | 2.77 pp | -28 | 48 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 560 | 265 | 295 | 47.32% | 50.42% | 48.33% | 2.68 pp | -30 | 48 | -0.62 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 734 | 351 | 383 | 47.82% | 47.08% | 49.79% | 2.18 pp | -32 | 43 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 911 | 437 | 474 | 47.97% | 51.25% | 48.12% | 2.03 pp | -37 | 48 | -0.77 |
| BTC Market Hours Daily | nn | NN | 560 | 260 | 300 | 46.43% | 45.83% | 47.92% | 3.57 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 560 | 259 | 301 | 46.25% | 49.17% | 46.67% | 3.75 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 911 | 430 | 481 | 47.20% | 47.50% | 46.46% | 2.80 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 734 | 340 | 394 | 46.32% | 44.58% | 47.29% | 3.68 pp | -54 | 43 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 506 | 219 | 287 | 43.28% | 42.08% | 43.33% | 6.72 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 506 | 217 | 289 | 42.89% | 43.75% | 43.12% | 7.11 pp | -72 | 48 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 506 | 208 | 298 | 41.11% | 42.08% | 41.67% | 8.89 pp | -90 | 48 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 560 | 231 | 329 | 41.25% | 41.67% | 40.42% | 8.75 pp | -98 | 48 | -2.04 |
| BTC Hourly | nn | NN | 911 | 405 | 506 | 44.46% | 43.75% | 42.08% | 5.54 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 911 | 405 | 506 | 44.46% | 44.17% | 43.96% | 5.54 pp | -101 | 48 | -2.10 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 560 | 226 | 334 | 40.36% | 38.75% | 40.42% | 9.64 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 560 | 224 | 336 | 40.00% | 40.83% | 39.17% | 10.00 pp | -112 | 48 | -2.33 |
| BTC Daily | lstm | LSTM | 734 | 315 | 419 | 42.92% | 36.67% | 41.25% | 7.08 pp | -104 | 43 | -2.42 |
| BTC Daily | rf | RandomForest | 734 | 312 | 422 | 42.51% | 40.42% | 43.33% | 7.49 pp | -110 | 43 | -2.56 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 911 | 390 | 521 | 42.81% | 39.58% | 41.67% | 7.19 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 911 | 382 | 529 | 41.93% | 40.83% | 40.62% | 8.07 pp | -147 | 48 | -3.06 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 744 | 294 | 450 | 39.52% | 36.25% | 38.12% | 10.48 pp | -156 | 43 | -3.63 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 911 | 437 | 474 | 47.97% | 51.25% | 48.12% | 2.03 pp | -37 | 48 | -0.77 |
| BTC Hourly | transformer | Transformer | 911 | 430 | 481 | 47.20% | 47.50% | 46.46% | 2.80 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 911 | 405 | 506 | 44.46% | 43.75% | 42.08% | 5.54 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 911 | 405 | 506 | 44.46% | 44.17% | 43.96% | 5.54 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 911 | 390 | 521 | 42.81% | 39.58% | 41.67% | 7.19 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 911 | 382 | 529 | 41.93% | 40.83% | 40.62% | 8.07 pp | -147 | 48 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 734 | 356 | 378 | 48.50% | 47.50% | 48.54% | 1.50 pp | -22 | 43 | -0.51 |
| BTC Daily | transformer | Transformer | 734 | 351 | 383 | 47.82% | 47.08% | 49.79% | 2.18 pp | -32 | 43 | -0.74 |
| BTC Daily | nn | NN | 734 | 340 | 394 | 46.32% | 44.58% | 47.29% | 3.68 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 734 | 315 | 419 | 42.92% | 36.67% | 41.25% | 7.08 pp | -104 | 43 | -2.42 |
| BTC Daily | rf | RandomForest | 734 | 312 | 422 | 42.51% | 40.42% | 43.33% | 7.49 pp | -110 | 43 | -2.56 |
| BTC Daily | xgb | XGBoost | 744 | 294 | 450 | 39.52% | 36.25% | 38.12% | 10.48 pp | -156 | 43 | -3.63 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 506 | 244 | 262 | 48.22% | 44.58% | 48.12% | 1.78 pp | -18 | 48 | -0.38 |
| BTC Market Hours | transformer | Transformer | 506 | 242 | 264 | 47.83% | 46.25% | 48.33% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Market Hours | nn | NN | 506 | 239 | 267 | 47.23% | 50.00% | 48.12% | 2.77 pp | -28 | 48 | -0.58 |
| BTC Market Hours | lstm | LSTM | 506 | 219 | 287 | 43.28% | 42.08% | 43.33% | 6.72 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 506 | 217 | 289 | 42.89% | 43.75% | 43.12% | 7.11 pp | -72 | 48 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 506 | 208 | 298 | 41.11% | 42.08% | 41.67% | 8.89 pp | -90 | 48 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 560 | 265 | 295 | 47.32% | 50.42% | 48.33% | 2.68 pp | -30 | 48 | -0.62 |
| BTC Market Hours Daily | nn | NN | 560 | 260 | 300 | 46.43% | 45.83% | 47.92% | 3.57 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 560 | 259 | 301 | 46.25% | 49.17% | 46.67% | 3.75 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 560 | 231 | 329 | 41.25% | 41.67% | 40.42% | 8.75 pp | -98 | 48 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 560 | 226 | 334 | 40.36% | 38.75% | 40.42% | 9.64 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 560 | 224 | 336 | 40.00% | 40.83% | 39.17% | 10.00 pp | -112 | 48 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 11 | -2.18 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
