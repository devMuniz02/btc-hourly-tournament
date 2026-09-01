# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T20:28:06.304134+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1060 | 695 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 19:00:00+00:00 | 697 | 457 | 239 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 19:00:00+00:00 | 699 | 511 | 186 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T00:00:00+00:00 | 107 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T00:00:00+00:00 | 107 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T00:00:00+00:00 | 107 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T00:00:00+00:00 | 108 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 10 | 5 | 66.67% | 66.67% | 66.67% | 16.67 pp | 5 | 2 | 2.50 |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | rf | RandomForest | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 457 | 221 | 236 | 48.36% | 44.17% | 48.36% | 1.64 pp | -15 | 45 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 685 | 335 | 350 | 48.91% | 46.67% | 49.58% | 1.09 pp | -15 | 41 | -0.37 |
| BTC Market Hours | nn | NN | 457 | 215 | 242 | 47.05% | 48.75% | 47.05% | 2.95 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 685 | 330 | 355 | 48.18% | 46.25% | 49.58% | 1.82 pp | -25 | 41 | -0.61 |
| Consolidated Hourly | lstm | LSTM | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 457 | 210 | 247 | 45.95% | 39.58% | 45.95% | 4.05 pp | -37 | 45 | -0.82 |
| BTC Market Hours Daily | nn | NN | 511 | 234 | 277 | 45.79% | 43.33% | 46.67% | 4.21 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 511 | 233 | 278 | 45.60% | 45.83% | 46.04% | 4.40 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 862 | 407 | 455 | 47.22% | 45.83% | 47.08% | 2.78 pp | -48 | 46 | -1.04 |
| BTC Daily | nn | NN | 685 | 320 | 365 | 46.72% | 42.92% | 48.96% | 3.28 pp | -45 | 41 | -1.10 |
| Consolidated Hourly | nn | NN | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | transformer | Transformer | 511 | 231 | 280 | 45.21% | 46.25% | 45.62% | 4.79 pp | -49 | 44 | -1.11 |
| BTC Hourly | transformer | Transformer | 862 | 405 | 457 | 46.98% | 47.08% | 46.67% | 3.02 pp | -52 | 46 | -1.13 |
| BTC Market Hours | rf | RandomForest | 457 | 198 | 259 | 43.33% | 43.33% | 43.33% | 6.67 pp | -61 | 45 | -1.36 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 6 | 9 | 40.00% | 40.00% | 40.00% | 10.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 457 | 194 | 263 | 42.45% | 40.00% | 42.45% | 7.55 pp | -69 | 45 | -1.53 |
| BTC Hourly | nn | NN | 862 | 389 | 473 | 45.13% | 46.25% | 44.38% | 4.87 pp | -84 | 46 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 457 | 186 | 271 | 40.70% | 39.17% | 40.70% | 9.30 pp | -85 | 45 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 511 | 212 | 299 | 41.49% | 41.67% | 41.67% | 8.51 pp | -87 | 44 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 862 | 383 | 479 | 44.43% | 43.33% | 43.75% | 5.57 pp | -96 | 46 | -2.09 |
| BTC Daily | lstm | LSTM | 685 | 298 | 387 | 43.50% | 38.33% | 42.50% | 6.50 pp | -89 | 41 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 511 | 205 | 306 | 40.12% | 38.33% | 40.83% | 9.88 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 685 | 294 | 391 | 42.92% | 40.83% | 43.33% | 7.08 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 511 | 202 | 309 | 39.53% | 36.25% | 39.17% | 10.47 pp | -107 | 44 | -2.43 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 862 | 367 | 495 | 42.58% | 37.92% | 41.88% | 7.42 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 862 | 364 | 498 | 42.23% | 40.42% | 42.92% | 7.77 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 695 | 275 | 420 | 39.57% | 35.00% | 39.38% | 10.43 pp | -145 | 41 | -3.54 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 685 | 335 | 350 | 48.91% | 46.67% | 49.58% | 1.09 pp | -15 | 41 | -0.37 |
| BTC Daily | transformer | Transformer | 685 | 330 | 355 | 48.18% | 46.25% | 49.58% | 1.82 pp | -25 | 41 | -0.61 |
| BTC Daily | nn | NN | 685 | 320 | 365 | 46.72% | 42.92% | 48.96% | 3.28 pp | -45 | 41 | -1.10 |
| BTC Daily | lstm | LSTM | 685 | 298 | 387 | 43.50% | 38.33% | 42.50% | 6.50 pp | -89 | 41 | -2.17 |
| BTC Daily | rf | RandomForest | 685 | 294 | 391 | 42.92% | 40.83% | 43.33% | 7.08 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 695 | 275 | 420 | 39.57% | 35.00% | 39.38% | 10.43 pp | -145 | 41 | -3.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 457 | 221 | 236 | 48.36% | 44.17% | 48.36% | 1.64 pp | -15 | 45 | -0.33 |
| BTC Market Hours | nn | NN | 457 | 215 | 242 | 47.05% | 48.75% | 47.05% | 2.95 pp | -27 | 45 | -0.60 |
| BTC Market Hours | transformer | Transformer | 457 | 210 | 247 | 45.95% | 39.58% | 45.95% | 4.05 pp | -37 | 45 | -0.82 |
| BTC Market Hours | rf | RandomForest | 457 | 198 | 259 | 43.33% | 43.33% | 43.33% | 6.67 pp | -61 | 45 | -1.36 |
| BTC Market Hours | lstm | LSTM | 457 | 194 | 263 | 42.45% | 40.00% | 42.45% | 7.55 pp | -69 | 45 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 457 | 186 | 271 | 40.70% | 39.17% | 40.70% | 9.30 pp | -85 | 45 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 511 | 234 | 277 | 45.79% | 43.33% | 46.67% | 4.21 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 511 | 233 | 278 | 45.60% | 45.83% | 46.04% | 4.40 pp | -45 | 44 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 511 | 231 | 280 | 45.21% | 46.25% | 45.62% | 4.79 pp | -49 | 44 | -1.11 |
| BTC Market Hours Daily | rf | RandomForest | 511 | 212 | 299 | 41.49% | 41.67% | 41.67% | 8.51 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 511 | 205 | 306 | 40.12% | 38.33% | 40.83% | 9.88 pp | -101 | 44 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 511 | 202 | 309 | 39.53% | 36.25% | 39.17% | 10.47 pp | -107 | 44 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | rf | RandomForest | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | xgb | XGBoost | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | nn | NN | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 10 | 5 | 66.67% | 66.67% | 66.67% | 16.67 pp | 5 | 2 | 2.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 6 | 9 | 40.00% | 40.00% | 40.00% | 10.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
