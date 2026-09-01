# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T18:27:40.984986+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1183 | 895 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1059 | 694 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 17:00:00+00:00 | 694 | 456 | 237 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 17:00:00+00:00 | 696 | 510 | 184 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 684 | 335 | 349 | 48.98% | 46.67% | 49.58% | 1.02 pp | -14 | 41 | -0.34 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 456 | 220 | 236 | 48.25% | 44.17% | 48.25% | 1.75 pp | -16 | 44 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Daily | transformer | Transformer | 684 | 330 | 354 | 48.25% | 46.25% | 49.58% | 1.75 pp | -24 | 41 | -0.59 |
| BTC Market Hours | nn | NN | 456 | 215 | 241 | 47.15% | 49.17% | 47.15% | 2.85 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 456 | 209 | 247 | 45.83% | 39.58% | 45.83% | 4.17 pp | -38 | 44 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 510 | 234 | 276 | 45.88% | 43.75% | 46.88% | 4.12 pp | -42 | 44 | -0.95 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 510 | 233 | 277 | 45.69% | 45.83% | 46.25% | 4.31 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 861 | 406 | 455 | 47.15% | 45.42% | 46.88% | 2.85 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 684 | 320 | 364 | 46.78% | 42.92% | 48.96% | 3.22 pp | -44 | 41 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 510 | 231 | 279 | 45.29% | 46.25% | 45.83% | 4.71 pp | -48 | 44 | -1.09 |
| BTC Hourly | transformer | Transformer | 861 | 404 | 457 | 46.92% | 46.67% | 46.46% | 3.08 pp | -53 | 46 | -1.15 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| BTC Market Hours | rf | RandomForest | 456 | 197 | 259 | 43.20% | 43.33% | 43.20% | 6.80 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 456 | 194 | 262 | 42.54% | 40.42% | 42.54% | 7.46 pp | -68 | 44 | -1.55 |
| BTC Hourly | nn | NN | 861 | 388 | 473 | 45.06% | 45.83% | 44.38% | 4.94 pp | -85 | 46 | -1.85 |
| BTC Market Hours | xgb | XGBoost | 456 | 185 | 271 | 40.57% | 39.17% | 40.57% | 9.43 pp | -86 | 44 | -1.95 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 510 | 211 | 299 | 41.37% | 41.25% | 41.46% | 8.63 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 861 | 383 | 478 | 44.48% | 43.75% | 43.75% | 5.52 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 684 | 297 | 387 | 43.42% | 37.92% | 42.29% | 6.58 pp | -90 | 41 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 510 | 205 | 305 | 40.20% | 38.33% | 41.04% | 9.80 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 684 | 293 | 391 | 42.84% | 40.42% | 43.12% | 7.16 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 510 | 201 | 309 | 39.41% | 35.83% | 38.96% | 10.59 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 861 | 366 | 495 | 42.51% | 37.50% | 41.67% | 7.49 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 861 | 363 | 498 | 42.16% | 40.00% | 42.71% | 7.84 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 694 | 275 | 419 | 39.63% | 35.00% | 39.58% | 10.37 pp | -144 | 41 | -3.51 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 861 | 406 | 455 | 47.15% | 45.42% | 46.88% | 2.85 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 861 | 404 | 457 | 46.92% | 46.67% | 46.46% | 3.08 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 861 | 388 | 473 | 45.06% | 45.83% | 44.38% | 4.94 pp | -85 | 46 | -1.85 |
| BTC Hourly | rf | RandomForest | 861 | 383 | 478 | 44.48% | 43.75% | 43.75% | 5.52 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 861 | 366 | 495 | 42.51% | 37.50% | 41.67% | 7.49 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 861 | 363 | 498 | 42.16% | 40.00% | 42.71% | 7.84 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 684 | 335 | 349 | 48.98% | 46.67% | 49.58% | 1.02 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 684 | 330 | 354 | 48.25% | 46.25% | 49.58% | 1.75 pp | -24 | 41 | -0.59 |
| BTC Daily | nn | NN | 684 | 320 | 364 | 46.78% | 42.92% | 48.96% | 3.22 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 684 | 297 | 387 | 43.42% | 37.92% | 42.29% | 6.58 pp | -90 | 41 | -2.20 |
| BTC Daily | rf | RandomForest | 684 | 293 | 391 | 42.84% | 40.42% | 43.12% | 7.16 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 694 | 275 | 419 | 39.63% | 35.00% | 39.58% | 10.37 pp | -144 | 41 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 456 | 220 | 236 | 48.25% | 44.17% | 48.25% | 1.75 pp | -16 | 44 | -0.36 |
| BTC Market Hours | nn | NN | 456 | 215 | 241 | 47.15% | 49.17% | 47.15% | 2.85 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 456 | 209 | 247 | 45.83% | 39.58% | 45.83% | 4.17 pp | -38 | 44 | -0.86 |
| BTC Market Hours | rf | RandomForest | 456 | 197 | 259 | 43.20% | 43.33% | 43.20% | 6.80 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 456 | 194 | 262 | 42.54% | 40.42% | 42.54% | 7.46 pp | -68 | 44 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 456 | 185 | 271 | 40.57% | 39.17% | 40.57% | 9.43 pp | -86 | 44 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 510 | 234 | 276 | 45.88% | 43.75% | 46.88% | 4.12 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 510 | 233 | 277 | 45.69% | 45.83% | 46.25% | 4.31 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 510 | 231 | 279 | 45.29% | 46.25% | 45.83% | 4.71 pp | -48 | 44 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 510 | 211 | 299 | 41.37% | 41.25% | 41.46% | 8.63 pp | -88 | 44 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 510 | 205 | 305 | 40.20% | 38.33% | 41.04% | 9.80 pp | -100 | 44 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 510 | 201 | 309 | 39.41% | 35.83% | 38.96% | 10.59 pp | -108 | 44 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
