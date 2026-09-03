# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T17:37:10.615944+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1215 | 927 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1090 | 725 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 750 | 487 | 262 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 752 | 541 | 209 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 346 | 369 | 48.39% | 46.67% | 48.54% | 1.61 pp | -23 | 43 | -0.53 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 715 | 340 | 375 | 47.55% | 45.42% | 49.58% | 2.45 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 541 | 251 | 290 | 46.40% | 49.58% | 47.29% | 3.60 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 893 | 425 | 468 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 541 | 248 | 293 | 45.84% | 47.92% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 541 | 248 | 293 | 45.84% | 44.58% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 893 | 423 | 470 | 47.37% | 48.33% | 47.29% | 2.63 pp | -47 | 47 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 715 | 330 | 385 | 46.15% | 43.33% | 47.50% | 3.85 pp | -55 | 43 | -1.28 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |
| BTC Hourly | nn | NN | 893 | 399 | 494 | 44.68% | 45.42% | 42.50% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | rf | RandomForest | 893 | 399 | 494 | 44.68% | 45.83% | 44.17% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 541 | 223 | 318 | 41.22% | 41.67% | 41.46% | 8.78 pp | -95 | 47 | -2.02 |
| BTC Daily | lstm | LSTM | 715 | 309 | 406 | 43.22% | 37.92% | 42.50% | 6.78 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 541 | 217 | 324 | 40.11% | 37.92% | 40.62% | 9.89 pp | -107 | 47 | -2.28 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 541 | 215 | 326 | 39.74% | 39.58% | 39.58% | 10.26 pp | -111 | 47 | -2.36 |
| BTC Daily | rf | RandomForest | 715 | 305 | 410 | 42.66% | 40.83% | 43.33% | 7.34 pp | -105 | 43 | -2.44 |
| BTC Hourly | lstm | LSTM | 893 | 383 | 510 | 42.89% | 39.58% | 42.29% | 7.11 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 893 | 377 | 516 | 42.22% | 42.92% | 42.08% | 7.78 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 725 | 286 | 439 | 39.45% | 34.58% | 38.54% | 10.55 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 893 | 425 | 468 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 893 | 423 | 470 | 47.37% | 48.33% | 47.29% | 2.63 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 893 | 399 | 494 | 44.68% | 45.42% | 42.50% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | rf | RandomForest | 893 | 399 | 494 | 44.68% | 45.83% | 44.17% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 893 | 383 | 510 | 42.89% | 39.58% | 42.29% | 7.11 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 893 | 377 | 516 | 42.22% | 42.92% | 42.08% | 7.78 pp | -139 | 47 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 346 | 369 | 48.39% | 46.67% | 48.54% | 1.61 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 715 | 340 | 375 | 47.55% | 45.42% | 49.58% | 2.45 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 715 | 330 | 385 | 46.15% | 43.33% | 47.50% | 3.85 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 715 | 309 | 406 | 43.22% | 37.92% | 42.50% | 6.78 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 715 | 305 | 410 | 42.66% | 40.83% | 43.33% | 7.34 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 725 | 286 | 439 | 39.45% | 34.58% | 38.54% | 10.55 pp | -153 | 43 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 541 | 251 | 290 | 46.40% | 49.58% | 47.29% | 3.60 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 541 | 248 | 293 | 45.84% | 47.92% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 541 | 248 | 293 | 45.84% | 44.58% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 541 | 223 | 318 | 41.22% | 41.67% | 41.46% | 8.78 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 541 | 217 | 324 | 40.11% | 37.92% | 40.62% | 9.89 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 541 | 215 | 326 | 39.74% | 39.58% | 39.58% | 10.26 pp | -111 | 47 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
