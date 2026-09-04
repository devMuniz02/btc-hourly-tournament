# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T18:11:45.322328+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1231 | 943 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1107 | 742 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 781 | 504 | 276 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 782 | 557 | 223 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 504 | 242 | 262 | 48.02% | 44.58% | 47.92% | 1.98 pp | -20 | 48 | -0.42 |
| BTC Market Hours | transformer | Transformer | 504 | 240 | 264 | 47.62% | 46.25% | 48.12% | 2.38 pp | -24 | 48 | -0.50 |
| BTC Market Hours | nn | NN | 504 | 239 | 265 | 47.42% | 50.83% | 48.33% | 2.58 pp | -26 | 48 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 732 | 354 | 378 | 48.36% | 47.08% | 48.12% | 1.64 pp | -24 | 43 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 557 | 262 | 295 | 47.04% | 49.58% | 47.92% | 2.96 pp | -33 | 48 | -0.69 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 909 | 436 | 473 | 47.96% | 51.25% | 48.33% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Daily | transformer | Transformer | 732 | 349 | 383 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 557 | 258 | 299 | 46.32% | 50.00% | 46.67% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 557 | 258 | 299 | 46.32% | 45.42% | 47.50% | 3.68 pp | -41 | 48 | -0.85 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 909 | 429 | 480 | 47.19% | 47.50% | 46.67% | 2.81 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 732 | 339 | 393 | 46.31% | 45.00% | 47.08% | 3.69 pp | -54 | 43 | -1.26 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 504 | 217 | 287 | 43.06% | 41.25% | 43.12% | 6.94 pp | -70 | 48 | -1.46 |
| BTC Market Hours | rf | RandomForest | 504 | 216 | 288 | 42.86% | 44.17% | 43.12% | 7.14 pp | -72 | 48 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 504 | 208 | 296 | 41.27% | 42.50% | 41.88% | 8.73 pp | -88 | 48 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 557 | 231 | 326 | 41.47% | 42.08% | 40.42% | 8.53 pp | -95 | 48 | -1.98 |
| BTC Hourly | nn | NN | 909 | 404 | 505 | 44.44% | 43.75% | 42.08% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 909 | 404 | 505 | 44.44% | 44.17% | 43.96% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 557 | 225 | 332 | 40.39% | 38.75% | 40.42% | 9.61 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 557 | 222 | 335 | 39.86% | 40.42% | 38.75% | 10.14 pp | -113 | 48 | -2.35 |
| BTC Daily | lstm | LSTM | 732 | 315 | 417 | 43.03% | 37.08% | 41.25% | 6.97 pp | -102 | 43 | -2.37 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 732 | 312 | 420 | 42.62% | 40.83% | 43.33% | 7.38 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 909 | 389 | 520 | 42.79% | 39.58% | 41.88% | 7.21 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 909 | 381 | 528 | 41.91% | 40.83% | 40.83% | 8.09 pp | -147 | 48 | -3.06 |
| BTC Daily | xgb | XGBoost | 742 | 293 | 449 | 39.49% | 35.83% | 38.12% | 10.51 pp | -156 | 43 | -3.63 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 909 | 436 | 473 | 47.96% | 51.25% | 48.33% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Hourly | transformer | Transformer | 909 | 429 | 480 | 47.19% | 47.50% | 46.67% | 2.81 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 909 | 404 | 505 | 44.44% | 43.75% | 42.08% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 909 | 404 | 505 | 44.44% | 44.17% | 43.96% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 909 | 389 | 520 | 42.79% | 39.58% | 41.88% | 7.21 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 909 | 381 | 528 | 41.91% | 40.83% | 40.83% | 8.09 pp | -147 | 48 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 732 | 354 | 378 | 48.36% | 47.08% | 48.12% | 1.64 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 732 | 349 | 383 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 732 | 339 | 393 | 46.31% | 45.00% | 47.08% | 3.69 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 732 | 315 | 417 | 43.03% | 37.08% | 41.25% | 6.97 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 732 | 312 | 420 | 42.62% | 40.83% | 43.33% | 7.38 pp | -108 | 43 | -2.51 |
| BTC Daily | xgb | XGBoost | 742 | 293 | 449 | 39.49% | 35.83% | 38.12% | 10.51 pp | -156 | 43 | -3.63 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 504 | 242 | 262 | 48.02% | 44.58% | 47.92% | 1.98 pp | -20 | 48 | -0.42 |
| BTC Market Hours | transformer | Transformer | 504 | 240 | 264 | 47.62% | 46.25% | 48.12% | 2.38 pp | -24 | 48 | -0.50 |
| BTC Market Hours | nn | NN | 504 | 239 | 265 | 47.42% | 50.83% | 48.33% | 2.58 pp | -26 | 48 | -0.54 |
| BTC Market Hours | lstm | LSTM | 504 | 217 | 287 | 43.06% | 41.25% | 43.12% | 6.94 pp | -70 | 48 | -1.46 |
| BTC Market Hours | rf | RandomForest | 504 | 216 | 288 | 42.86% | 44.17% | 43.12% | 7.14 pp | -72 | 48 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 504 | 208 | 296 | 41.27% | 42.50% | 41.88% | 8.73 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 557 | 262 | 295 | 47.04% | 49.58% | 47.92% | 2.96 pp | -33 | 48 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 557 | 258 | 299 | 46.32% | 50.00% | 46.67% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 557 | 258 | 299 | 46.32% | 45.42% | 47.50% | 3.68 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 557 | 231 | 326 | 41.47% | 42.08% | 40.42% | 8.53 pp | -95 | 48 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 557 | 225 | 332 | 40.39% | 38.75% | 40.42% | 9.61 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 557 | 222 | 335 | 39.86% | 40.42% | 38.75% | 10.14 pp | -113 | 48 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
