# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T10:35:15.715043+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1178 | 890 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1054 | 689 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 683 | 451 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 684 | 504 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 103 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 103 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 11 | 92 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 11 | 92 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 451 | 219 | 232 | 48.56% | 45.00% | 48.56% | 1.44 pp | -13 | 44 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 679 | 333 | 346 | 49.04% | 47.92% | 49.79% | 0.96 pp | -13 | 41 | -0.32 |
| Consolidated Hourly | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| BTC Daily | transformer | Transformer | 679 | 328 | 351 | 48.31% | 46.67% | 49.38% | 1.69 pp | -23 | 41 | -0.56 |
| BTC Market Hours | nn | NN | 451 | 212 | 239 | 47.01% | 47.92% | 47.01% | 2.99 pp | -27 | 44 | -0.61 |
| Consolidated Hourly | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 451 | 208 | 243 | 46.12% | 40.00% | 46.12% | 3.88 pp | -35 | 44 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 504 | 231 | 273 | 45.83% | 46.25% | 46.04% | 4.17 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 504 | 230 | 274 | 45.63% | 43.33% | 46.46% | 4.37 pp | -44 | 44 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 679 | 318 | 361 | 46.83% | 43.33% | 49.17% | 3.17 pp | -43 | 41 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 856 | 403 | 453 | 47.08% | 45.00% | 46.88% | 2.92 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 856 | 403 | 453 | 47.08% | 47.08% | 46.88% | 2.92 pp | -50 | 46 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 504 | 228 | 276 | 45.24% | 45.83% | 45.21% | 4.76 pp | -48 | 44 | -1.09 |
| BTC Market Hours | rf | RandomForest | 451 | 195 | 256 | 43.24% | 43.33% | 43.24% | 6.76 pp | -61 | 44 | -1.39 |
| BTC Market Hours | lstm | LSTM | 451 | 191 | 260 | 42.35% | 39.58% | 42.35% | 7.65 pp | -69 | 44 | -1.57 |
| Consolidated Hourly | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| BTC Hourly | nn | NN | 856 | 386 | 470 | 45.09% | 45.00% | 44.17% | 4.91 pp | -84 | 46 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 504 | 209 | 295 | 41.47% | 41.67% | 41.67% | 8.53 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 451 | 181 | 270 | 40.13% | 37.92% | 40.13% | 9.87 pp | -89 | 44 | -2.02 |
| BTC Hourly | rf | RandomForest | 856 | 381 | 475 | 44.51% | 43.33% | 43.96% | 5.49 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 679 | 296 | 383 | 43.59% | 38.75% | 42.71% | 6.41 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 679 | 292 | 387 | 43.00% | 40.83% | 43.75% | 7.00 pp | -95 | 41 | -2.32 |
| BTC Market Hours Daily | lstm | LSTM | 504 | 201 | 303 | 39.88% | 37.50% | 40.42% | 10.12 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 504 | 197 | 307 | 39.09% | 36.25% | 38.54% | 10.91 pp | -110 | 44 | -2.50 |
| BTC Hourly | lstm | LSTM | 856 | 364 | 492 | 42.52% | 37.92% | 41.88% | 7.48 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 856 | 360 | 496 | 42.06% | 40.00% | 42.29% | 7.94 pp | -136 | 46 | -2.96 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 689 | 274 | 415 | 39.77% | 35.00% | 39.58% | 10.23 pp | -141 | 41 | -3.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 856 | 403 | 453 | 47.08% | 45.00% | 46.88% | 2.92 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 856 | 403 | 453 | 47.08% | 47.08% | 46.88% | 2.92 pp | -50 | 46 | -1.09 |
| BTC Hourly | nn | NN | 856 | 386 | 470 | 45.09% | 45.00% | 44.17% | 4.91 pp | -84 | 46 | -1.83 |
| BTC Hourly | rf | RandomForest | 856 | 381 | 475 | 44.51% | 43.33% | 43.96% | 5.49 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 856 | 364 | 492 | 42.52% | 37.92% | 41.88% | 7.48 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 856 | 360 | 496 | 42.06% | 40.00% | 42.29% | 7.94 pp | -136 | 46 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 679 | 333 | 346 | 49.04% | 47.92% | 49.79% | 0.96 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 679 | 328 | 351 | 48.31% | 46.67% | 49.38% | 1.69 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 679 | 318 | 361 | 46.83% | 43.33% | 49.17% | 3.17 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 679 | 296 | 383 | 43.59% | 38.75% | 42.71% | 6.41 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 679 | 292 | 387 | 43.00% | 40.83% | 43.75% | 7.00 pp | -95 | 41 | -2.32 |
| BTC Daily | xgb | XGBoost | 689 | 274 | 415 | 39.77% | 35.00% | 39.58% | 10.23 pp | -141 | 41 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 451 | 219 | 232 | 48.56% | 45.00% | 48.56% | 1.44 pp | -13 | 44 | -0.30 |
| BTC Market Hours | nn | NN | 451 | 212 | 239 | 47.01% | 47.92% | 47.01% | 2.99 pp | -27 | 44 | -0.61 |
| BTC Market Hours | transformer | Transformer | 451 | 208 | 243 | 46.12% | 40.00% | 46.12% | 3.88 pp | -35 | 44 | -0.80 |
| BTC Market Hours | rf | RandomForest | 451 | 195 | 256 | 43.24% | 43.33% | 43.24% | 6.76 pp | -61 | 44 | -1.39 |
| BTC Market Hours | lstm | LSTM | 451 | 191 | 260 | 42.35% | 39.58% | 42.35% | 7.65 pp | -69 | 44 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 451 | 181 | 270 | 40.13% | 37.92% | 40.13% | 9.87 pp | -89 | 44 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 504 | 231 | 273 | 45.83% | 46.25% | 46.04% | 4.17 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 504 | 230 | 274 | 45.63% | 43.33% | 46.46% | 4.37 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 504 | 228 | 276 | 45.24% | 45.83% | 45.21% | 4.76 pp | -48 | 44 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 504 | 209 | 295 | 41.47% | 41.67% | 41.67% | 8.53 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 504 | 201 | 303 | 39.88% | 37.50% | 40.42% | 10.12 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 504 | 197 | 307 | 39.09% | 36.25% | 38.54% | 10.91 pp | -110 | 44 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
