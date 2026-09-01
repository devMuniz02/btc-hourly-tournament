# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T17:05:36.654997+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1182 | 894 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1058 | 693 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 16:00:00+00:00 | 692 | 455 | 236 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 16:00:00+00:00 | 694 | 509 | 183 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T23:00:00+00:00 | 106 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T23:00:00+00:00 | 106 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T23:00:00+00:00 | 106 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T23:00:00+00:00 | 107 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | rf | RandomForest | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 683 | 335 | 348 | 49.05% | 47.08% | 49.58% | 0.95 pp | -13 | 41 | -0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 455 | 220 | 235 | 48.35% | 44.17% | 48.35% | 1.65 pp | -15 | 44 | -0.34 |
| Consolidated Hourly | xgb | XGBoost | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Daily | transformer | Transformer | 683 | 330 | 353 | 48.32% | 46.25% | 49.58% | 1.68 pp | -23 | 41 | -0.56 |
| BTC Market Hours | nn | NN | 455 | 215 | 240 | 47.25% | 49.17% | 47.25% | 2.75 pp | -25 | 44 | -0.57 |
| BTC Market Hours | transformer | Transformer | 455 | 209 | 246 | 45.93% | 39.58% | 45.93% | 4.07 pp | -37 | 44 | -0.84 |
| Consolidated Hourly | lstm | LSTM | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 509 | 233 | 276 | 45.78% | 45.83% | 46.25% | 4.22 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 509 | 233 | 276 | 45.78% | 43.33% | 46.67% | 4.22 pp | -43 | 44 | -0.98 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 860 | 406 | 454 | 47.21% | 45.42% | 46.88% | 2.79 pp | -48 | 46 | -1.04 |
| BTC Daily | nn | NN | 683 | 320 | 363 | 46.85% | 43.33% | 49.17% | 3.15 pp | -43 | 41 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 509 | 231 | 278 | 45.38% | 46.25% | 45.83% | 4.62 pp | -47 | 44 | -1.07 |
| Consolidated Hourly | nn | NN | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| BTC Hourly | transformer | Transformer | 860 | 404 | 456 | 46.98% | 47.08% | 46.46% | 3.02 pp | -52 | 46 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| BTC Market Hours | rf | RandomForest | 455 | 196 | 259 | 43.08% | 42.92% | 43.08% | 6.92 pp | -63 | 44 | -1.43 |
| BTC Market Hours | lstm | LSTM | 455 | 194 | 261 | 42.64% | 40.42% | 42.64% | 7.36 pp | -67 | 44 | -1.52 |
| BTC Hourly | nn | NN | 860 | 387 | 473 | 45.00% | 45.42% | 44.17% | 5.00 pp | -86 | 46 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 509 | 211 | 298 | 41.45% | 41.25% | 41.46% | 8.55 pp | -87 | 44 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 455 | 184 | 271 | 40.44% | 38.75% | 40.44% | 9.56 pp | -87 | 44 | -1.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 860 | 383 | 477 | 44.53% | 43.75% | 43.75% | 5.47 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 683 | 297 | 386 | 43.48% | 38.33% | 42.29% | 6.52 pp | -89 | 41 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 509 | 204 | 305 | 40.08% | 37.92% | 40.83% | 9.92 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 683 | 293 | 390 | 42.90% | 40.83% | 43.12% | 7.10 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 509 | 200 | 309 | 39.29% | 35.42% | 38.75% | 10.71 pp | -109 | 44 | -2.48 |
| BTC Hourly | lstm | LSTM | 860 | 365 | 495 | 42.44% | 37.50% | 41.46% | 7.56 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 860 | 363 | 497 | 42.21% | 40.00% | 42.71% | 7.79 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 693 | 275 | 418 | 39.68% | 35.00% | 39.58% | 10.32 pp | -143 | 41 | -3.49 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 860 | 406 | 454 | 47.21% | 45.42% | 46.88% | 2.79 pp | -48 | 46 | -1.04 |
| BTC Hourly | transformer | Transformer | 860 | 404 | 456 | 46.98% | 47.08% | 46.46% | 3.02 pp | -52 | 46 | -1.13 |
| BTC Hourly | nn | NN | 860 | 387 | 473 | 45.00% | 45.42% | 44.17% | 5.00 pp | -86 | 46 | -1.87 |
| BTC Hourly | rf | RandomForest | 860 | 383 | 477 | 44.53% | 43.75% | 43.75% | 5.47 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 860 | 365 | 495 | 42.44% | 37.50% | 41.46% | 7.56 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 860 | 363 | 497 | 42.21% | 40.00% | 42.71% | 7.79 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 683 | 335 | 348 | 49.05% | 47.08% | 49.58% | 0.95 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 683 | 330 | 353 | 48.32% | 46.25% | 49.58% | 1.68 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 683 | 320 | 363 | 46.85% | 43.33% | 49.17% | 3.15 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 683 | 297 | 386 | 43.48% | 38.33% | 42.29% | 6.52 pp | -89 | 41 | -2.17 |
| BTC Daily | rf | RandomForest | 683 | 293 | 390 | 42.90% | 40.83% | 43.12% | 7.10 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 693 | 275 | 418 | 39.68% | 35.00% | 39.58% | 10.32 pp | -143 | 41 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 455 | 220 | 235 | 48.35% | 44.17% | 48.35% | 1.65 pp | -15 | 44 | -0.34 |
| BTC Market Hours | nn | NN | 455 | 215 | 240 | 47.25% | 49.17% | 47.25% | 2.75 pp | -25 | 44 | -0.57 |
| BTC Market Hours | transformer | Transformer | 455 | 209 | 246 | 45.93% | 39.58% | 45.93% | 4.07 pp | -37 | 44 | -0.84 |
| BTC Market Hours | rf | RandomForest | 455 | 196 | 259 | 43.08% | 42.92% | 43.08% | 6.92 pp | -63 | 44 | -1.43 |
| BTC Market Hours | lstm | LSTM | 455 | 194 | 261 | 42.64% | 40.42% | 42.64% | 7.36 pp | -67 | 44 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 455 | 184 | 271 | 40.44% | 38.75% | 40.44% | 9.56 pp | -87 | 44 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 509 | 233 | 276 | 45.78% | 45.83% | 46.25% | 4.22 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 509 | 233 | 276 | 45.78% | 43.33% | 46.67% | 4.22 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 509 | 231 | 278 | 45.38% | 46.25% | 45.83% | 4.62 pp | -47 | 44 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 509 | 211 | 298 | 41.45% | 41.25% | 41.46% | 8.55 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 509 | 204 | 305 | 40.08% | 37.92% | 40.83% | 9.92 pp | -101 | 44 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 509 | 200 | 309 | 39.29% | 35.42% | 38.75% | 10.71 pp | -109 | 44 | -2.48 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | rf | RandomForest | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | lstm | LSTM | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | nn | NN | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| Consolidated Hourly | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 8 | 6 | 57.14% | 57.14% | 57.14% | 7.14 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 6 | 8 | 42.86% | 42.86% | 42.86% | 7.14 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
