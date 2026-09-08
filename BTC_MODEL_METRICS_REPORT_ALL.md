# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T06:37:15.644234+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1288 | 1000 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1164 | 799 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 884 | 561 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 886 | 615 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 204 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 561 | 274 | 287 | 48.84% | 47.50% | 47.92% | 1.16 pp | -13 | 53 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 561 | 267 | 294 | 47.59% | 51.67% | 49.38% | 2.41 pp | -27 | 53 | -0.51 |
| BTC Daily | mlp_sklearn | MLPClassifier | 789 | 381 | 408 | 48.29% | 47.08% | 48.12% | 1.71 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 561 | 264 | 297 | 47.06% | 46.67% | 47.50% | 2.94 pp | -33 | 53 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 615 | 287 | 328 | 46.67% | 49.17% | 47.29% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | nn | NN | 615 | 287 | 328 | 46.67% | 47.50% | 48.33% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 615 | 287 | 328 | 46.67% | 48.75% | 47.50% | 3.33 pp | -41 | 52 | -0.79 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 966 | 460 | 506 | 47.62% | 49.58% | 46.67% | 2.38 pp | -46 | 50 | -0.92 |
| BTC Daily | transformer | Transformer | 789 | 367 | 422 | 46.51% | 40.42% | 46.46% | 3.49 pp | -55 | 46 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 789 | 366 | 423 | 46.39% | 45.00% | 45.00% | 3.61 pp | -57 | 46 | -1.24 |
| BTC Hourly | transformer | Transformer | 966 | 450 | 516 | 46.58% | 44.58% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | lstm | LSTM | 561 | 244 | 317 | 43.49% | 42.50% | 43.96% | 6.51 pp | -73 | 53 | -1.38 |
| BTC Market Hours | rf | RandomForest | 561 | 243 | 318 | 43.32% | 45.83% | 43.54% | 6.68 pp | -75 | 53 | -1.42 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 561 | 239 | 322 | 42.60% | 46.25% | 42.92% | 7.40 pp | -83 | 53 | -1.57 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 615 | 258 | 357 | 41.95% | 44.58% | 41.25% | 8.05 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 615 | 251 | 364 | 40.81% | 43.33% | 40.62% | 9.19 pp | -113 | 52 | -2.17 |
| BTC Hourly | rf | RandomForest | 966 | 428 | 538 | 44.31% | 42.08% | 42.71% | 5.69 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 615 | 250 | 365 | 40.65% | 40.83% | 40.21% | 9.35 pp | -115 | 52 | -2.21 |
| BTC Hourly | nn | NN | 966 | 427 | 539 | 44.20% | 41.67% | 42.50% | 5.80 pp | -112 | 50 | -2.24 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 789 | 332 | 457 | 42.08% | 34.17% | 40.00% | 7.92 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 789 | 330 | 459 | 41.83% | 38.33% | 41.67% | 8.17 pp | -129 | 46 | -2.80 |
| BTC Hourly | lstm | LSTM | 966 | 412 | 554 | 42.65% | 37.92% | 41.67% | 7.35 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 966 | 399 | 567 | 41.30% | 36.67% | 38.75% | 8.70 pp | -168 | 50 | -3.36 |
| BTC Daily | xgb | XGBoost | 799 | 312 | 487 | 39.05% | 35.83% | 36.25% | 10.95 pp | -175 | 46 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 966 | 460 | 506 | 47.62% | 49.58% | 46.67% | 2.38 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 966 | 450 | 516 | 46.58% | 44.58% | 43.96% | 3.42 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 966 | 428 | 538 | 44.31% | 42.08% | 42.71% | 5.69 pp | -110 | 50 | -2.20 |
| BTC Hourly | nn | NN | 966 | 427 | 539 | 44.20% | 41.67% | 42.50% | 5.80 pp | -112 | 50 | -2.24 |
| BTC Hourly | lstm | LSTM | 966 | 412 | 554 | 42.65% | 37.92% | 41.67% | 7.35 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 966 | 399 | 567 | 41.30% | 36.67% | 38.75% | 8.70 pp | -168 | 50 | -3.36 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 789 | 381 | 408 | 48.29% | 47.08% | 48.12% | 1.71 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 789 | 367 | 422 | 46.51% | 40.42% | 46.46% | 3.49 pp | -55 | 46 | -1.20 |
| BTC Daily | nn | NN | 789 | 366 | 423 | 46.39% | 45.00% | 45.00% | 3.61 pp | -57 | 46 | -1.24 |
| BTC Daily | lstm | LSTM | 789 | 332 | 457 | 42.08% | 34.17% | 40.00% | 7.92 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 789 | 330 | 459 | 41.83% | 38.33% | 41.67% | 8.17 pp | -129 | 46 | -2.80 |
| BTC Daily | xgb | XGBoost | 799 | 312 | 487 | 39.05% | 35.83% | 36.25% | 10.95 pp | -175 | 46 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 561 | 274 | 287 | 48.84% | 47.50% | 47.92% | 1.16 pp | -13 | 53 | -0.25 |
| BTC Market Hours | nn | NN | 561 | 267 | 294 | 47.59% | 51.67% | 49.38% | 2.41 pp | -27 | 53 | -0.51 |
| BTC Market Hours | transformer | Transformer | 561 | 264 | 297 | 47.06% | 46.67% | 47.50% | 2.94 pp | -33 | 53 | -0.62 |
| BTC Market Hours | lstm | LSTM | 561 | 244 | 317 | 43.49% | 42.50% | 43.96% | 6.51 pp | -73 | 53 | -1.38 |
| BTC Market Hours | rf | RandomForest | 561 | 243 | 318 | 43.32% | 45.83% | 43.54% | 6.68 pp | -75 | 53 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 561 | 239 | 322 | 42.60% | 46.25% | 42.92% | 7.40 pp | -83 | 53 | -1.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 615 | 287 | 328 | 46.67% | 49.17% | 47.29% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | nn | NN | 615 | 287 | 328 | 46.67% | 47.50% | 48.33% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 615 | 287 | 328 | 46.67% | 48.75% | 47.50% | 3.33 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 615 | 258 | 357 | 41.95% | 44.58% | 41.25% | 8.05 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 615 | 251 | 364 | 40.81% | 43.33% | 40.62% | 9.19 pp | -113 | 52 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 615 | 250 | 365 | 40.65% | 40.83% | 40.21% | 9.35 pp | -115 | 52 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
