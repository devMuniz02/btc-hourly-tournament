# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T15:55:16.337706+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1326 | 1038 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1202 | 837 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 951 | 599 | 351 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 953 | 653 | 298 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 599 | 289 | 310 | 48.25% | 46.25% | 47.29% | 1.75 pp | -21 | 55 | -0.38 |
| BTC Market Hours | nn | NN | 599 | 287 | 312 | 47.91% | 51.25% | 49.58% | 2.09 pp | -25 | 55 | -0.45 |
| BTC Market Hours | transformer | Transformer | 599 | 281 | 318 | 46.91% | 46.25% | 46.25% | 3.09 pp | -37 | 55 | -0.67 |
| BTC Market Hours Daily | nn | NN | 653 | 305 | 348 | 46.71% | 47.92% | 47.50% | 3.29 pp | -43 | 55 | -0.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 827 | 395 | 432 | 47.76% | 45.00% | 46.25% | 2.24 pp | -37 | 47 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 653 | 304 | 349 | 46.55% | 47.92% | 47.08% | 3.45 pp | -45 | 55 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 653 | 304 | 349 | 46.55% | 47.92% | 47.92% | 3.45 pp | -45 | 55 | -0.82 |
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1004 | 475 | 529 | 47.31% | 47.92% | 45.83% | 2.69 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 827 | 386 | 441 | 46.67% | 45.42% | 45.62% | 3.33 pp | -55 | 47 | -1.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1004 | 468 | 536 | 46.61% | 46.25% | 45.00% | 3.39 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 827 | 382 | 445 | 46.19% | 37.92% | 45.00% | 3.81 pp | -63 | 47 | -1.34 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 599 | 257 | 342 | 42.90% | 42.50% | 42.71% | 7.10 pp | -85 | 55 | -1.55 |
| BTC Market Hours | rf | RandomForest | 599 | 255 | 344 | 42.57% | 42.08% | 42.08% | 7.43 pp | -89 | 55 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 599 | 255 | 344 | 42.57% | 45.00% | 43.33% | 7.43 pp | -89 | 55 | -1.62 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 653 | 269 | 384 | 41.19% | 41.67% | 40.83% | 8.81 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 653 | 268 | 385 | 41.04% | 42.92% | 40.62% | 8.96 pp | -117 | 55 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 653 | 268 | 385 | 41.04% | 42.50% | 40.62% | 8.96 pp | -117 | 55 | -2.13 |
| BTC Hourly | nn | NN | 1004 | 443 | 561 | 44.12% | 42.50% | 41.46% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1004 | 443 | 561 | 44.12% | 42.08% | 43.54% | 5.88 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 827 | 348 | 479 | 42.08% | 35.00% | 40.00% | 7.92 pp | -131 | 47 | -2.79 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 1004 | 425 | 579 | 42.33% | 36.67% | 40.00% | 7.67 pp | -154 | 52 | -2.96 |
| BTC Daily | rf | RandomForest | 827 | 343 | 484 | 41.48% | 37.08% | 40.42% | 8.52 pp | -141 | 47 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1004 | 414 | 590 | 41.24% | 36.25% | 38.96% | 8.76 pp | -176 | 52 | -3.38 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 837 | 329 | 508 | 39.31% | 36.67% | 36.46% | 10.69 pp | -179 | 47 | -3.81 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1004 | 475 | 529 | 47.31% | 47.92% | 45.83% | 2.69 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1004 | 468 | 536 | 46.61% | 46.25% | 45.00% | 3.39 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1004 | 443 | 561 | 44.12% | 42.50% | 41.46% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1004 | 443 | 561 | 44.12% | 42.08% | 43.54% | 5.88 pp | -118 | 52 | -2.27 |
| BTC Hourly | lstm | LSTM | 1004 | 425 | 579 | 42.33% | 36.67% | 40.00% | 7.67 pp | -154 | 52 | -2.96 |
| BTC Hourly | xgb | XGBoost | 1004 | 414 | 590 | 41.24% | 36.25% | 38.96% | 8.76 pp | -176 | 52 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 827 | 395 | 432 | 47.76% | 45.00% | 46.25% | 2.24 pp | -37 | 47 | -0.79 |
| BTC Daily | nn | NN | 827 | 386 | 441 | 46.67% | 45.42% | 45.62% | 3.33 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 827 | 382 | 445 | 46.19% | 37.92% | 45.00% | 3.81 pp | -63 | 47 | -1.34 |
| BTC Daily | lstm | LSTM | 827 | 348 | 479 | 42.08% | 35.00% | 40.00% | 7.92 pp | -131 | 47 | -2.79 |
| BTC Daily | rf | RandomForest | 827 | 343 | 484 | 41.48% | 37.08% | 40.42% | 8.52 pp | -141 | 47 | -3.00 |
| BTC Daily | xgb | XGBoost | 837 | 329 | 508 | 39.31% | 36.67% | 36.46% | 10.69 pp | -179 | 47 | -3.81 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 599 | 289 | 310 | 48.25% | 46.25% | 47.29% | 1.75 pp | -21 | 55 | -0.38 |
| BTC Market Hours | nn | NN | 599 | 287 | 312 | 47.91% | 51.25% | 49.58% | 2.09 pp | -25 | 55 | -0.45 |
| BTC Market Hours | transformer | Transformer | 599 | 281 | 318 | 46.91% | 46.25% | 46.25% | 3.09 pp | -37 | 55 | -0.67 |
| BTC Market Hours | lstm | LSTM | 599 | 257 | 342 | 42.90% | 42.50% | 42.71% | 7.10 pp | -85 | 55 | -1.55 |
| BTC Market Hours | rf | RandomForest | 599 | 255 | 344 | 42.57% | 42.08% | 42.08% | 7.43 pp | -89 | 55 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 599 | 255 | 344 | 42.57% | 45.00% | 43.33% | 7.43 pp | -89 | 55 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 653 | 305 | 348 | 46.71% | 47.92% | 47.50% | 3.29 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 653 | 304 | 349 | 46.55% | 47.92% | 47.08% | 3.45 pp | -45 | 55 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 653 | 304 | 349 | 46.55% | 47.92% | 47.92% | 3.45 pp | -45 | 55 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 653 | 269 | 384 | 41.19% | 41.67% | 40.83% | 8.81 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 653 | 268 | 385 | 41.04% | 42.92% | 40.62% | 8.96 pp | -117 | 55 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 653 | 268 | 385 | 41.04% | 42.50% | 40.62% | 8.96 pp | -117 | 55 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
