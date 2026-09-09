# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T10:35:09.956662+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1307 | 1019 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1183 | 818 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 916 | 580 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 918 | 634 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 223 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 580 | 281 | 299 | 48.45% | 46.67% | 47.29% | 1.55 pp | -18 | 54 | -0.33 |
| BTC Market Hours | nn | NN | 580 | 279 | 301 | 48.10% | 52.50% | 49.79% | 1.90 pp | -22 | 54 | -0.41 |
| BTC Market Hours | transformer | Transformer | 580 | 275 | 305 | 47.41% | 48.33% | 47.29% | 2.59 pp | -30 | 54 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 808 | 389 | 419 | 48.14% | 46.67% | 47.08% | 1.86 pp | -30 | 46 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 634 | 299 | 335 | 47.16% | 50.42% | 47.71% | 2.84 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | nn | NN | 634 | 298 | 336 | 47.00% | 47.92% | 48.12% | 3.00 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 634 | 296 | 338 | 46.69% | 48.33% | 46.88% | 3.31 pp | -42 | 54 | -0.78 |
| Consolidated Hourly | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 985 | 470 | 515 | 47.72% | 50.83% | 46.67% | 2.28 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| BTC Daily | nn | NN | 808 | 375 | 433 | 46.41% | 43.75% | 45.21% | 3.59 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 808 | 374 | 434 | 46.29% | 39.58% | 46.25% | 3.71 pp | -60 | 46 | -1.30 |
| BTC Hourly | transformer | Transformer | 985 | 456 | 529 | 46.29% | 44.17% | 43.75% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 580 | 250 | 330 | 43.10% | 45.83% | 43.54% | 6.90 pp | -80 | 54 | -1.48 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours | lstm | LSTM | 580 | 249 | 331 | 42.93% | 41.67% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |
| BTC Market Hours | rf | RandomForest | 580 | 249 | 331 | 42.93% | 44.17% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |
| Consolidated Hourly | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 634 | 263 | 371 | 41.48% | 42.50% | 40.42% | 8.52 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 634 | 262 | 372 | 41.32% | 44.58% | 40.62% | 8.68 pp | -110 | 54 | -2.04 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 634 | 258 | 376 | 40.69% | 40.83% | 40.00% | 9.31 pp | -118 | 54 | -2.19 |
| BTC Hourly | rf | RandomForest | 985 | 435 | 550 | 44.16% | 41.67% | 42.92% | 5.84 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| BTC Hourly | nn | NN | 985 | 434 | 551 | 44.06% | 41.67% | 42.08% | 5.94 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 808 | 340 | 468 | 42.08% | 34.58% | 40.42% | 7.92 pp | -128 | 46 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 985 | 419 | 566 | 42.54% | 37.50% | 40.62% | 7.46 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 808 | 335 | 473 | 41.46% | 36.25% | 41.25% | 8.54 pp | -138 | 46 | -3.00 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 985 | 406 | 579 | 41.22% | 35.83% | 38.75% | 8.78 pp | -173 | 51 | -3.39 |
| BTC Daily | xgb | XGBoost | 818 | 319 | 499 | 39.00% | 35.00% | 35.62% | 11.00 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 985 | 470 | 515 | 47.72% | 50.83% | 46.67% | 2.28 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 985 | 456 | 529 | 46.29% | 44.17% | 43.75% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 985 | 435 | 550 | 44.16% | 41.67% | 42.92% | 5.84 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 985 | 434 | 551 | 44.06% | 41.67% | 42.08% | 5.94 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 985 | 419 | 566 | 42.54% | 37.50% | 40.62% | 7.46 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 985 | 406 | 579 | 41.22% | 35.83% | 38.75% | 8.78 pp | -173 | 51 | -3.39 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 808 | 389 | 419 | 48.14% | 46.67% | 47.08% | 1.86 pp | -30 | 46 | -0.65 |
| BTC Daily | nn | NN | 808 | 375 | 433 | 46.41% | 43.75% | 45.21% | 3.59 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 808 | 374 | 434 | 46.29% | 39.58% | 46.25% | 3.71 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 808 | 340 | 468 | 42.08% | 34.58% | 40.42% | 7.92 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 808 | 335 | 473 | 41.46% | 36.25% | 41.25% | 8.54 pp | -138 | 46 | -3.00 |
| BTC Daily | xgb | XGBoost | 818 | 319 | 499 | 39.00% | 35.00% | 35.62% | 11.00 pp | -180 | 46 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 580 | 281 | 299 | 48.45% | 46.67% | 47.29% | 1.55 pp | -18 | 54 | -0.33 |
| BTC Market Hours | nn | NN | 580 | 279 | 301 | 48.10% | 52.50% | 49.79% | 1.90 pp | -22 | 54 | -0.41 |
| BTC Market Hours | transformer | Transformer | 580 | 275 | 305 | 47.41% | 48.33% | 47.29% | 2.59 pp | -30 | 54 | -0.56 |
| BTC Market Hours | xgb | XGBoost | 580 | 250 | 330 | 43.10% | 45.83% | 43.54% | 6.90 pp | -80 | 54 | -1.48 |
| BTC Market Hours | lstm | LSTM | 580 | 249 | 331 | 42.93% | 41.67% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |
| BTC Market Hours | rf | RandomForest | 580 | 249 | 331 | 42.93% | 44.17% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 634 | 299 | 335 | 47.16% | 50.42% | 47.71% | 2.84 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | nn | NN | 634 | 298 | 336 | 47.00% | 47.92% | 48.12% | 3.00 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 634 | 296 | 338 | 46.69% | 48.33% | 46.88% | 3.31 pp | -42 | 54 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 634 | 263 | 371 | 41.48% | 42.50% | 40.42% | 8.52 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 634 | 262 | 372 | 41.32% | 44.58% | 40.62% | 8.68 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 634 | 258 | 376 | 40.69% | 40.83% | 40.00% | 9.31 pp | -118 | 54 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
