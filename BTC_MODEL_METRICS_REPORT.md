# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T05:37:02.506527+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1222 | 934 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1098 | 733 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 766 | 495 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 768 | 549 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 144 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 17 | 17 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 495 | 239 | 256 | 48.28% | 44.58% | 48.12% | 1.72 pp | -17 | 47 | -0.36 |
| BTC Market Hours | nn | NN | 495 | 234 | 261 | 47.27% | 50.42% | 47.71% | 2.73 pp | -27 | 47 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 723 | 349 | 374 | 48.27% | 46.67% | 47.92% | 1.73 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 495 | 232 | 263 | 46.87% | 43.75% | 47.50% | 3.13 pp | -31 | 47 | -0.66 |
| BTC Daily | transformer | Transformer | 723 | 346 | 377 | 47.86% | 46.67% | 50.21% | 2.14 pp | -31 | 43 | -0.72 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 900 | 431 | 469 | 47.89% | 51.25% | 48.54% | 2.11 pp | -38 | 47 | -0.81 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 549 | 255 | 294 | 46.45% | 49.17% | 47.29% | 3.55 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 549 | 253 | 296 | 46.08% | 45.00% | 47.29% | 3.92 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 549 | 252 | 297 | 45.90% | 48.75% | 46.88% | 4.10 pp | -45 | 47 | -0.96 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 900 | 426 | 474 | 47.33% | 48.33% | 46.88% | 2.67 pp | -48 | 47 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 723 | 336 | 387 | 46.47% | 45.00% | 47.71% | 3.53 pp | -51 | 43 | -1.19 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 495 | 214 | 281 | 43.23% | 41.25% | 43.33% | 6.77 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 495 | 214 | 281 | 43.23% | 44.17% | 43.54% | 6.77 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 495 | 203 | 292 | 41.01% | 40.83% | 41.04% | 8.99 pp | -89 | 47 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 549 | 229 | 320 | 41.71% | 42.50% | 41.25% | 8.29 pp | -91 | 47 | -1.94 |
| BTC Hourly | nn | NN | 900 | 401 | 499 | 44.56% | 44.58% | 42.29% | 5.44 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 900 | 400 | 500 | 44.44% | 44.58% | 44.17% | 5.56 pp | -100 | 47 | -2.13 |
| Consolidated Hourly | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 549 | 221 | 328 | 40.26% | 38.33% | 40.62% | 9.74 pp | -107 | 47 | -2.28 |
| BTC Daily | lstm | LSTM | 723 | 312 | 411 | 43.15% | 37.50% | 41.88% | 6.85 pp | -99 | 43 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 549 | 220 | 329 | 40.07% | 41.25% | 39.38% | 9.93 pp | -109 | 47 | -2.32 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 723 | 310 | 413 | 42.88% | 41.67% | 43.75% | 7.12 pp | -103 | 43 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 900 | 385 | 515 | 42.78% | 39.58% | 42.29% | 7.22 pp | -130 | 47 | -2.77 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 900 | 378 | 522 | 42.00% | 42.08% | 41.67% | 8.00 pp | -144 | 47 | -3.06 |
| BTC Daily | xgb | XGBoost | 733 | 292 | 441 | 39.84% | 37.08% | 38.96% | 10.16 pp | -149 | 43 | -3.47 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 900 | 431 | 469 | 47.89% | 51.25% | 48.54% | 2.11 pp | -38 | 47 | -0.81 |
| BTC Hourly | transformer | Transformer | 900 | 426 | 474 | 47.33% | 48.33% | 46.88% | 2.67 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 900 | 401 | 499 | 44.56% | 44.58% | 42.29% | 5.44 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 900 | 400 | 500 | 44.44% | 44.58% | 44.17% | 5.56 pp | -100 | 47 | -2.13 |
| BTC Hourly | lstm | LSTM | 900 | 385 | 515 | 42.78% | 39.58% | 42.29% | 7.22 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 900 | 378 | 522 | 42.00% | 42.08% | 41.67% | 8.00 pp | -144 | 47 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 723 | 349 | 374 | 48.27% | 46.67% | 47.92% | 1.73 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 723 | 346 | 377 | 47.86% | 46.67% | 50.21% | 2.14 pp | -31 | 43 | -0.72 |
| BTC Daily | nn | NN | 723 | 336 | 387 | 46.47% | 45.00% | 47.71% | 3.53 pp | -51 | 43 | -1.19 |
| BTC Daily | lstm | LSTM | 723 | 312 | 411 | 43.15% | 37.50% | 41.88% | 6.85 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 723 | 310 | 413 | 42.88% | 41.67% | 43.75% | 7.12 pp | -103 | 43 | -2.40 |
| BTC Daily | xgb | XGBoost | 733 | 292 | 441 | 39.84% | 37.08% | 38.96% | 10.16 pp | -149 | 43 | -3.47 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 495 | 239 | 256 | 48.28% | 44.58% | 48.12% | 1.72 pp | -17 | 47 | -0.36 |
| BTC Market Hours | nn | NN | 495 | 234 | 261 | 47.27% | 50.42% | 47.71% | 2.73 pp | -27 | 47 | -0.57 |
| BTC Market Hours | transformer | Transformer | 495 | 232 | 263 | 46.87% | 43.75% | 47.50% | 3.13 pp | -31 | 47 | -0.66 |
| BTC Market Hours | lstm | LSTM | 495 | 214 | 281 | 43.23% | 41.25% | 43.33% | 6.77 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 495 | 214 | 281 | 43.23% | 44.17% | 43.54% | 6.77 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 495 | 203 | 292 | 41.01% | 40.83% | 41.04% | 8.99 pp | -89 | 47 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 549 | 255 | 294 | 46.45% | 49.17% | 47.29% | 3.55 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 549 | 253 | 296 | 46.08% | 45.00% | 47.29% | 3.92 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 549 | 252 | 297 | 45.90% | 48.75% | 46.88% | 4.10 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 549 | 229 | 320 | 41.71% | 42.50% | 41.25% | 8.29 pp | -91 | 47 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 549 | 221 | 328 | 40.26% | 38.33% | 40.62% | 9.74 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 549 | 220 | 329 | 40.07% | 41.25% | 39.38% | 9.93 pp | -109 | 47 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 17 | 17 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
