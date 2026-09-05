# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T18:27:44.331936+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 810 | 520 | 289 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 812 | 574 | 236 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 520 | 253 | 267 | 48.65% | 45.83% | 48.75% | 1.35 pp | -14 | 49 | -0.29 |
| BTC Market Hours | transformer | Transformer | 520 | 250 | 270 | 48.08% | 47.92% | 48.54% | 1.92 pp | -20 | 49 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 748 | 365 | 383 | 48.80% | 48.33% | 49.17% | 1.20 pp | -18 | 44 | -0.41 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 574 | 274 | 300 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 49 | -0.53 |
| BTC Market Hours | nn | NN | 520 | 246 | 274 | 47.31% | 50.83% | 48.75% | 2.69 pp | -28 | 49 | -0.57 |
| BTC Market Hours Daily | nn | NN | 574 | 268 | 306 | 46.69% | 46.25% | 48.12% | 3.31 pp | -38 | 49 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 925 | 442 | 483 | 47.78% | 49.58% | 47.29% | 2.22 pp | -41 | 49 | -0.84 |
| BTC Daily | transformer | Transformer | 748 | 355 | 393 | 47.46% | 45.00% | 48.96% | 2.54 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 574 | 264 | 310 | 45.99% | 49.17% | 46.67% | 4.01 pp | -46 | 49 | -0.94 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 925 | 436 | 489 | 47.14% | 46.67% | 45.62% | 2.86 pp | -53 | 49 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Daily | nn | NN | 748 | 348 | 400 | 46.52% | 44.17% | 47.08% | 3.48 pp | -52 | 44 | -1.18 |
| BTC Market Hours | lstm | LSTM | 520 | 227 | 293 | 43.65% | 43.33% | 44.17% | 6.35 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 520 | 225 | 295 | 43.27% | 45.83% | 43.75% | 6.73 pp | -70 | 49 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 520 | 215 | 305 | 41.35% | 43.33% | 42.08% | 8.65 pp | -90 | 49 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 574 | 240 | 334 | 41.81% | 43.33% | 41.25% | 8.19 pp | -94 | 49 | -1.92 |
| BTC Hourly | rf | RandomForest | 925 | 414 | 511 | 44.76% | 44.58% | 44.38% | 5.24 pp | -97 | 49 | -1.98 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| BTC Hourly | nn | NN | 925 | 411 | 514 | 44.43% | 42.92% | 42.29% | 5.57 pp | -103 | 49 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 574 | 235 | 339 | 40.94% | 40.83% | 40.83% | 9.06 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 574 | 229 | 345 | 39.90% | 41.67% | 39.38% | 10.10 pp | -116 | 49 | -2.37 |
| BTC Daily | lstm | LSTM | 748 | 319 | 429 | 42.65% | 35.83% | 40.83% | 7.35 pp | -110 | 44 | -2.50 |
| BTC Daily | rf | RandomForest | 748 | 315 | 433 | 42.11% | 38.33% | 42.08% | 7.89 pp | -118 | 44 | -2.68 |
| BTC Hourly | lstm | LSTM | 925 | 396 | 529 | 42.81% | 37.92% | 41.88% | 7.19 pp | -133 | 49 | -2.71 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 925 | 387 | 538 | 41.84% | 39.58% | 40.42% | 8.16 pp | -151 | 49 | -3.08 |
| BTC Daily | xgb | XGBoost | 758 | 298 | 460 | 39.31% | 36.25% | 37.29% | 10.69 pp | -162 | 44 | -3.68 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 925 | 442 | 483 | 47.78% | 49.58% | 47.29% | 2.22 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 925 | 436 | 489 | 47.14% | 46.67% | 45.62% | 2.86 pp | -53 | 49 | -1.08 |
| BTC Hourly | rf | RandomForest | 925 | 414 | 511 | 44.76% | 44.58% | 44.38% | 5.24 pp | -97 | 49 | -1.98 |
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
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
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
