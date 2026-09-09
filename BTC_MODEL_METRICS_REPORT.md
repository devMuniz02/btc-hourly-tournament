# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T11:27:13.926896+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 580 | 281 | 299 | 48.45% | 46.67% | 47.29% | 1.55 pp | -18 | 54 | -0.33 |
| BTC Market Hours | nn | NN | 580 | 279 | 301 | 48.10% | 52.50% | 49.79% | 1.90 pp | -22 | 54 | -0.41 |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 580 | 275 | 305 | 47.41% | 48.33% | 47.29% | 2.59 pp | -30 | 54 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 808 | 389 | 419 | 48.14% | 46.67% | 47.08% | 1.86 pp | -30 | 46 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 634 | 299 | 335 | 47.16% | 50.42% | 47.71% | 2.84 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | nn | NN | 634 | 298 | 336 | 47.00% | 47.92% | 48.12% | 3.00 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 634 | 296 | 338 | 46.69% | 48.33% | 46.88% | 3.31 pp | -42 | 54 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 985 | 470 | 515 | 47.72% | 50.83% | 46.67% | 2.28 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 808 | 375 | 433 | 46.41% | 43.75% | 45.21% | 3.59 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 808 | 374 | 434 | 46.29% | 39.58% | 46.25% | 3.71 pp | -60 | 46 | -1.30 |
| BTC Hourly | transformer | Transformer | 985 | 456 | 529 | 46.29% | 44.17% | 43.75% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 580 | 250 | 330 | 43.10% | 45.83% | 43.54% | 6.90 pp | -80 | 54 | -1.48 |
| BTC Market Hours | lstm | LSTM | 580 | 249 | 331 | 42.93% | 41.67% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |
| BTC Market Hours | rf | RandomForest | 580 | 249 | 331 | 42.93% | 44.17% | 43.12% | 7.07 pp | -82 | 54 | -1.52 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 634 | 263 | 371 | 41.48% | 42.50% | 40.42% | 8.52 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 634 | 262 | 372 | 41.32% | 44.58% | 40.62% | 8.68 pp | -110 | 54 | -2.04 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 634 | 258 | 376 | 40.69% | 40.83% | 40.00% | 9.31 pp | -118 | 54 | -2.19 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 985 | 436 | 549 | 44.26% | 42.08% | 43.12% | 5.74 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 985 | 435 | 550 | 44.16% | 42.08% | 42.29% | 5.84 pp | -115 | 51 | -2.25 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Daily | lstm | LSTM | 808 | 340 | 468 | 42.08% | 34.58% | 40.42% | 7.92 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 985 | 420 | 565 | 42.64% | 37.92% | 40.83% | 7.36 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 808 | 335 | 473 | 41.46% | 36.25% | 41.25% | 8.54 pp | -138 | 46 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 985 | 406 | 579 | 41.22% | 35.83% | 38.75% | 8.78 pp | -173 | 51 | -3.39 |
| BTC Daily | xgb | XGBoost | 818 | 319 | 499 | 39.00% | 35.00% | 35.62% | 11.00 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 985 | 470 | 515 | 47.72% | 50.83% | 46.67% | 2.28 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 985 | 456 | 529 | 46.29% | 44.17% | 43.75% | 3.71 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 985 | 436 | 549 | 44.26% | 42.08% | 43.12% | 5.74 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 985 | 435 | 550 | 44.16% | 42.08% | 42.29% | 5.84 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 985 | 420 | 565 | 42.64% | 37.92% | 40.83% | 7.36 pp | -145 | 51 | -2.84 |
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
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
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
