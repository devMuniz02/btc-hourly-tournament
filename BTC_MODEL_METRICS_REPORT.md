# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T19:44:49.375433+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1184 | 896 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1059 | 694 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 18:00:00+00:00 | 695 | 456 | 238 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 18:00:00+00:00 | 697 | 510 | 185 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 456 | 220 | 236 | 48.25% | 44.17% | 48.25% | 1.75 pp | -16 | 44 | -0.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 684 | 334 | 350 | 48.83% | 46.25% | 49.38% | 1.17 pp | -16 | 41 | -0.39 |
| BTC Market Hours | nn | NN | 456 | 215 | 241 | 47.15% | 49.17% | 47.15% | 2.85 pp | -26 | 44 | -0.59 |
| BTC Daily | transformer | Transformer | 684 | 329 | 355 | 48.10% | 45.83% | 49.38% | 1.90 pp | -26 | 41 | -0.63 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 456 | 209 | 247 | 45.83% | 39.58% | 45.83% | 4.17 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | nn | NN | 510 | 234 | 276 | 45.88% | 43.75% | 46.88% | 4.12 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 510 | 233 | 277 | 45.69% | 45.83% | 46.25% | 4.31 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 862 | 407 | 455 | 47.22% | 45.83% | 47.08% | 2.78 pp | -48 | 46 | -1.04 |
| BTC Market Hours Daily | transformer | Transformer | 510 | 231 | 279 | 45.29% | 46.25% | 45.83% | 4.71 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 684 | 319 | 365 | 46.64% | 42.50% | 48.75% | 3.36 pp | -46 | 41 | -1.12 |
| BTC Hourly | transformer | Transformer | 862 | 405 | 457 | 46.98% | 47.08% | 46.67% | 3.02 pp | -52 | 46 | -1.13 |
| BTC Market Hours | rf | RandomForest | 456 | 197 | 259 | 43.20% | 43.33% | 43.20% | 6.80 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 456 | 194 | 262 | 42.54% | 40.42% | 42.54% | 7.46 pp | -68 | 44 | -1.55 |
| BTC Hourly | nn | NN | 862 | 389 | 473 | 45.13% | 46.25% | 44.38% | 4.87 pp | -84 | 46 | -1.83 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 456 | 185 | 271 | 40.57% | 39.17% | 40.57% | 9.43 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 510 | 211 | 299 | 41.37% | 41.25% | 41.46% | 8.63 pp | -88 | 44 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 862 | 383 | 479 | 44.43% | 43.33% | 43.75% | 5.57 pp | -96 | 46 | -2.09 |
| BTC Daily | lstm | LSTM | 684 | 298 | 386 | 43.57% | 38.33% | 42.50% | 6.43 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 510 | 205 | 305 | 40.20% | 38.33% | 41.04% | 9.80 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 684 | 293 | 391 | 42.84% | 40.42% | 43.12% | 7.16 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 510 | 201 | 309 | 39.41% | 35.83% | 38.96% | 10.59 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 862 | 367 | 495 | 42.58% | 37.92% | 41.88% | 7.42 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 862 | 364 | 498 | 42.23% | 40.42% | 42.92% | 7.77 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 694 | 274 | 420 | 39.48% | 34.58% | 39.38% | 10.52 pp | -146 | 41 | -3.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 862 | 407 | 455 | 47.22% | 45.83% | 47.08% | 2.78 pp | -48 | 46 | -1.04 |
| BTC Hourly | transformer | Transformer | 862 | 405 | 457 | 46.98% | 47.08% | 46.67% | 3.02 pp | -52 | 46 | -1.13 |
| BTC Hourly | nn | NN | 862 | 389 | 473 | 45.13% | 46.25% | 44.38% | 4.87 pp | -84 | 46 | -1.83 |
| BTC Hourly | rf | RandomForest | 862 | 383 | 479 | 44.43% | 43.33% | 43.75% | 5.57 pp | -96 | 46 | -2.09 |
| BTC Hourly | lstm | LSTM | 862 | 367 | 495 | 42.58% | 37.92% | 41.88% | 7.42 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 862 | 364 | 498 | 42.23% | 40.42% | 42.92% | 7.77 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 684 | 334 | 350 | 48.83% | 46.25% | 49.38% | 1.17 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 684 | 329 | 355 | 48.10% | 45.83% | 49.38% | 1.90 pp | -26 | 41 | -0.63 |
| BTC Daily | nn | NN | 684 | 319 | 365 | 46.64% | 42.50% | 48.75% | 3.36 pp | -46 | 41 | -1.12 |
| BTC Daily | lstm | LSTM | 684 | 298 | 386 | 43.57% | 38.33% | 42.50% | 6.43 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 684 | 293 | 391 | 42.84% | 40.42% | 43.12% | 7.16 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 694 | 274 | 420 | 39.48% | 34.58% | 39.38% | 10.52 pp | -146 | 41 | -3.56 |

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
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
