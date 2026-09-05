# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T16:10:04.159825+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1246 | 958 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1122 | 757 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 15:00:00+00:00 | 807 | 519 | 287 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 15:00:00+00:00 | 809 | 573 | 234 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 519 | 252 | 267 | 48.55% | 45.83% | 48.75% | 1.45 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 519 | 250 | 269 | 48.17% | 47.92% | 48.54% | 1.83 pp | -19 | 49 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 747 | 364 | 383 | 48.73% | 47.92% | 49.17% | 1.27 pp | -19 | 44 | -0.43 |
| BTC Market Hours Daily | transformer | Transformer | 573 | 274 | 299 | 47.82% | 52.08% | 49.17% | 2.18 pp | -25 | 49 | -0.51 |
| BTC Market Hours | nn | NN | 519 | 246 | 273 | 47.40% | 50.83% | 48.75% | 2.60 pp | -27 | 49 | -0.55 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 573 | 267 | 306 | 46.60% | 46.25% | 47.92% | 3.40 pp | -39 | 49 | -0.80 |
| BTC Daily | transformer | Transformer | 747 | 355 | 392 | 47.52% | 45.42% | 49.17% | 2.48 pp | -37 | 44 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 924 | 441 | 483 | 47.73% | 49.17% | 47.08% | 2.27 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 573 | 263 | 310 | 45.90% | 49.17% | 46.46% | 4.10 pp | -47 | 49 | -0.96 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 924 | 436 | 488 | 47.19% | 46.67% | 45.62% | 2.81 pp | -52 | 48 | -1.08 |
| BTC Daily | nn | NN | 747 | 347 | 400 | 46.45% | 43.75% | 47.08% | 3.55 pp | -53 | 44 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 519 | 227 | 292 | 43.74% | 43.75% | 44.38% | 6.26 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 519 | 225 | 294 | 43.35% | 45.83% | 43.96% | 6.65 pp | -69 | 49 | -1.41 |
| Consolidated Hourly | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 519 | 215 | 304 | 41.43% | 43.75% | 42.29% | 8.57 pp | -89 | 49 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 573 | 239 | 334 | 41.71% | 42.92% | 41.04% | 8.29 pp | -95 | 49 | -1.94 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 924 | 412 | 512 | 44.59% | 43.75% | 44.17% | 5.41 pp | -100 | 48 | -2.08 |
| Consolidated Hourly | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| BTC Hourly | nn | NN | 924 | 411 | 513 | 44.48% | 42.92% | 42.50% | 5.52 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 573 | 234 | 339 | 40.84% | 40.42% | 40.83% | 9.16 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 573 | 228 | 345 | 39.79% | 41.25% | 39.17% | 10.21 pp | -117 | 49 | -2.39 |
| BTC Daily | lstm | LSTM | 747 | 319 | 428 | 42.70% | 35.83% | 40.83% | 7.30 pp | -109 | 44 | -2.48 |
| BTC Daily | rf | RandomForest | 747 | 315 | 432 | 42.17% | 38.33% | 42.29% | 7.83 pp | -117 | 44 | -2.66 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 924 | 395 | 529 | 42.75% | 37.92% | 41.67% | 7.25 pp | -134 | 48 | -2.79 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 924 | 387 | 537 | 41.88% | 39.58% | 40.42% | 8.12 pp | -150 | 48 | -3.12 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 757 | 298 | 459 | 39.37% | 36.25% | 37.29% | 10.63 pp | -161 | 44 | -3.66 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 924 | 441 | 483 | 47.73% | 49.17% | 47.08% | 2.27 pp | -42 | 48 | -0.88 |
| BTC Hourly | transformer | Transformer | 924 | 436 | 488 | 47.19% | 46.67% | 45.62% | 2.81 pp | -52 | 48 | -1.08 |
| BTC Hourly | rf | RandomForest | 924 | 412 | 512 | 44.59% | 43.75% | 44.17% | 5.41 pp | -100 | 48 | -2.08 |
| BTC Hourly | nn | NN | 924 | 411 | 513 | 44.48% | 42.92% | 42.50% | 5.52 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 924 | 395 | 529 | 42.75% | 37.92% | 41.67% | 7.25 pp | -134 | 48 | -2.79 |
| BTC Hourly | xgb | XGBoost | 924 | 387 | 537 | 41.88% | 39.58% | 40.42% | 8.12 pp | -150 | 48 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 747 | 364 | 383 | 48.73% | 47.92% | 49.17% | 1.27 pp | -19 | 44 | -0.43 |
| BTC Daily | transformer | Transformer | 747 | 355 | 392 | 47.52% | 45.42% | 49.17% | 2.48 pp | -37 | 44 | -0.84 |
| BTC Daily | nn | NN | 747 | 347 | 400 | 46.45% | 43.75% | 47.08% | 3.55 pp | -53 | 44 | -1.20 |
| BTC Daily | lstm | LSTM | 747 | 319 | 428 | 42.70% | 35.83% | 40.83% | 7.30 pp | -109 | 44 | -2.48 |
| BTC Daily | rf | RandomForest | 747 | 315 | 432 | 42.17% | 38.33% | 42.29% | 7.83 pp | -117 | 44 | -2.66 |
| BTC Daily | xgb | XGBoost | 757 | 298 | 459 | 39.37% | 36.25% | 37.29% | 10.63 pp | -161 | 44 | -3.66 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 519 | 252 | 267 | 48.55% | 45.83% | 48.75% | 1.45 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 519 | 250 | 269 | 48.17% | 47.92% | 48.54% | 1.83 pp | -19 | 49 | -0.39 |
| BTC Market Hours | nn | NN | 519 | 246 | 273 | 47.40% | 50.83% | 48.75% | 2.60 pp | -27 | 49 | -0.55 |
| BTC Market Hours | lstm | LSTM | 519 | 227 | 292 | 43.74% | 43.75% | 44.38% | 6.26 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 519 | 225 | 294 | 43.35% | 45.83% | 43.96% | 6.65 pp | -69 | 49 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 519 | 215 | 304 | 41.43% | 43.75% | 42.29% | 8.57 pp | -89 | 49 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 573 | 274 | 299 | 47.82% | 52.08% | 49.17% | 2.18 pp | -25 | 49 | -0.51 |
| BTC Market Hours Daily | nn | NN | 573 | 267 | 306 | 46.60% | 46.25% | 47.92% | 3.40 pp | -39 | 49 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 573 | 263 | 310 | 45.90% | 49.17% | 46.46% | 4.10 pp | -47 | 49 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 573 | 239 | 334 | 41.71% | 42.92% | 41.04% | 8.29 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 573 | 234 | 339 | 40.84% | 40.42% | 40.83% | 9.16 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 573 | 228 | 345 | 39.79% | 41.25% | 39.17% | 10.21 pp | -117 | 49 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
