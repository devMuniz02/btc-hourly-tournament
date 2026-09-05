# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T17:39:55.434494+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1247 | 959 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1123 | 758 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 16:00:00+00:00 | 809 | 520 | 288 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 16:00:00+00:00 | 811 | 574 | 235 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 167 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 520 | 253 | 267 | 48.65% | 45.83% | 48.75% | 1.35 pp | -14 | 49 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 520 | 250 | 270 | 48.08% | 47.92% | 48.54% | 1.92 pp | -20 | 49 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 748 | 365 | 383 | 48.80% | 48.33% | 49.17% | 1.20 pp | -18 | 44 | -0.41 |
| BTC Market Hours Daily | transformer | Transformer | 574 | 274 | 300 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 49 | -0.53 |
| BTC Market Hours | nn | NN | 520 | 246 | 274 | 47.31% | 50.83% | 48.75% | 2.69 pp | -28 | 49 | -0.57 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 574 | 268 | 306 | 46.69% | 46.25% | 48.12% | 3.31 pp | -38 | 49 | -0.78 |
| BTC Daily | transformer | Transformer | 748 | 355 | 393 | 47.46% | 45.00% | 48.96% | 2.54 pp | -38 | 44 | -0.86 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 925 | 441 | 484 | 47.68% | 49.17% | 47.08% | 2.32 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 574 | 264 | 310 | 45.99% | 49.17% | 46.67% | 4.01 pp | -46 | 49 | -0.94 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 925 | 436 | 489 | 47.14% | 46.67% | 45.62% | 2.86 pp | -53 | 49 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| BTC Daily | nn | NN | 748 | 348 | 400 | 46.52% | 44.17% | 47.08% | 3.48 pp | -52 | 44 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 520 | 227 | 293 | 43.65% | 43.33% | 44.17% | 6.35 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 520 | 225 | 295 | 43.27% | 45.83% | 43.75% | 6.73 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 520 | 215 | 305 | 41.35% | 43.33% | 42.08% | 8.65 pp | -90 | 49 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 574 | 240 | 334 | 41.81% | 43.33% | 41.25% | 8.19 pp | -94 | 49 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 925 | 413 | 512 | 44.65% | 44.17% | 44.17% | 5.35 pp | -99 | 49 | -2.02 |
| BTC Hourly | nn | NN | 925 | 411 | 514 | 44.43% | 42.92% | 42.29% | 5.57 pp | -103 | 49 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 574 | 235 | 339 | 40.94% | 40.83% | 40.83% | 9.06 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 574 | 229 | 345 | 39.90% | 41.67% | 39.38% | 10.10 pp | -116 | 49 | -2.37 |
| BTC Daily | lstm | LSTM | 748 | 319 | 429 | 42.65% | 35.83% | 40.83% | 7.35 pp | -110 | 44 | -2.50 |
| BTC Daily | rf | RandomForest | 748 | 315 | 433 | 42.11% | 38.33% | 42.08% | 7.89 pp | -118 | 44 | -2.68 |
| BTC Hourly | lstm | LSTM | 925 | 396 | 529 | 42.81% | 37.92% | 41.88% | 7.19 pp | -133 | 49 | -2.71 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 925 | 387 | 538 | 41.84% | 39.58% | 40.42% | 8.16 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 758 | 298 | 460 | 39.31% | 36.25% | 37.29% | 10.69 pp | -162 | 44 | -3.68 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 925 | 441 | 484 | 47.68% | 49.17% | 47.08% | 2.32 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 925 | 436 | 489 | 47.14% | 46.67% | 45.62% | 2.86 pp | -53 | 49 | -1.08 |
| BTC Hourly | rf | RandomForest | 925 | 413 | 512 | 44.65% | 44.17% | 44.17% | 5.35 pp | -99 | 49 | -2.02 |
| BTC Hourly | nn | NN | 925 | 411 | 514 | 44.43% | 42.92% | 42.29% | 5.57 pp | -103 | 49 | -2.10 |
| BTC Hourly | lstm | LSTM | 925 | 396 | 529 | 42.81% | 37.92% | 41.88% | 7.19 pp | -133 | 49 | -2.71 |
| BTC Hourly | xgb | XGBoost | 925 | 387 | 538 | 41.84% | 39.58% | 40.42% | 8.16 pp | -151 | 49 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 748 | 365 | 383 | 48.80% | 48.33% | 49.17% | 1.20 pp | -18 | 44 | -0.41 |
| BTC Daily | transformer | Transformer | 748 | 355 | 393 | 47.46% | 45.00% | 48.96% | 2.54 pp | -38 | 44 | -0.86 |
| BTC Daily | nn | NN | 748 | 348 | 400 | 46.52% | 44.17% | 47.08% | 3.48 pp | -52 | 44 | -1.18 |
| BTC Daily | lstm | LSTM | 748 | 319 | 429 | 42.65% | 35.83% | 40.83% | 7.35 pp | -110 | 44 | -2.50 |
| BTC Daily | rf | RandomForest | 748 | 315 | 433 | 42.11% | 38.33% | 42.08% | 7.89 pp | -118 | 44 | -2.68 |
| BTC Daily | xgb | XGBoost | 758 | 298 | 460 | 39.31% | 36.25% | 37.29% | 10.69 pp | -162 | 44 | -3.68 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 520 | 253 | 267 | 48.65% | 45.83% | 48.75% | 1.35 pp | -14 | 49 | -0.29 |
| BTC Market Hours | transformer | Transformer | 520 | 250 | 270 | 48.08% | 47.92% | 48.54% | 1.92 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 520 | 246 | 274 | 47.31% | 50.83% | 48.75% | 2.69 pp | -28 | 49 | -0.57 |
| BTC Market Hours | lstm | LSTM | 520 | 227 | 293 | 43.65% | 43.33% | 44.17% | 6.35 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 520 | 225 | 295 | 43.27% | 45.83% | 43.75% | 6.73 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 520 | 215 | 305 | 41.35% | 43.33% | 42.08% | 8.65 pp | -90 | 49 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 574 | 274 | 300 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 49 | -0.53 |
| BTC Market Hours Daily | nn | NN | 574 | 268 | 306 | 46.69% | 46.25% | 48.12% | 3.31 pp | -38 | 49 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 574 | 264 | 310 | 45.99% | 49.17% | 46.67% | 4.01 pp | -46 | 49 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 574 | 240 | 334 | 41.81% | 43.33% | 41.25% | 8.19 pp | -94 | 49 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 574 | 235 | 339 | 40.94% | 40.83% | 40.83% | 9.06 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 574 | 229 | 345 | 39.90% | 41.67% | 39.38% | 10.10 pp | -116 | 49 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |

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
