# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T16:35:40.959212+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1057 | 692 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 15:00:00+00:00 | 690 | 454 | 235 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 15:00:00+00:00 | 692 | 508 | 182 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 454 | 220 | 234 | 48.46% | 44.58% | 48.46% | 1.54 pp | -14 | 44 | -0.32 |
| BTC Daily | mlp_sklearn | MLPClassifier | 682 | 334 | 348 | 48.97% | 47.08% | 49.38% | 1.03 pp | -14 | 41 | -0.34 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Daily | transformer | Transformer | 682 | 329 | 353 | 48.24% | 46.25% | 49.58% | 1.76 pp | -24 | 41 | -0.59 |
| BTC Market Hours | nn | NN | 454 | 214 | 240 | 47.14% | 48.75% | 47.14% | 2.86 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 454 | 209 | 245 | 46.04% | 39.58% | 46.04% | 3.96 pp | -36 | 44 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 508 | 233 | 275 | 45.87% | 43.33% | 46.88% | 4.13 pp | -42 | 44 | -0.95 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 508 | 232 | 276 | 45.67% | 45.42% | 46.25% | 4.33 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 860 | 406 | 454 | 47.21% | 45.42% | 46.88% | 2.79 pp | -48 | 46 | -1.04 |
| BTC Market Hours Daily | transformer | Transformer | 508 | 231 | 277 | 45.47% | 46.67% | 45.83% | 4.53 pp | -46 | 44 | -1.05 |
| BTC Daily | nn | NN | 682 | 319 | 363 | 46.77% | 43.33% | 48.96% | 3.23 pp | -44 | 41 | -1.07 |
| BTC Hourly | transformer | Transformer | 860 | 404 | 456 | 46.98% | 47.08% | 46.46% | 3.02 pp | -52 | 46 | -1.13 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| BTC Market Hours | rf | RandomForest | 454 | 196 | 258 | 43.17% | 42.92% | 43.17% | 6.83 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 454 | 193 | 261 | 42.51% | 40.42% | 42.51% | 7.49 pp | -68 | 44 | -1.55 |
| BTC Hourly | nn | NN | 860 | 387 | 473 | 45.00% | 45.42% | 44.17% | 5.00 pp | -86 | 46 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 508 | 211 | 297 | 41.54% | 41.25% | 41.46% | 8.46 pp | -86 | 44 | -1.95 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 454 | 183 | 271 | 40.31% | 38.75% | 40.31% | 9.69 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 860 | 383 | 477 | 44.53% | 43.75% | 43.75% | 5.47 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 682 | 297 | 385 | 43.55% | 38.33% | 42.50% | 6.45 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 508 | 203 | 305 | 39.96% | 37.50% | 40.83% | 10.04 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 682 | 292 | 390 | 42.82% | 40.42% | 43.12% | 7.18 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 508 | 199 | 309 | 39.17% | 35.42% | 38.54% | 10.83 pp | -110 | 44 | -2.50 |
| BTC Hourly | lstm | LSTM | 860 | 365 | 495 | 42.44% | 37.50% | 41.46% | 7.56 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 860 | 363 | 497 | 42.21% | 40.00% | 42.71% | 7.79 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 692 | 274 | 418 | 39.60% | 34.58% | 39.38% | 10.40 pp | -144 | 41 | -3.51 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 682 | 334 | 348 | 48.97% | 47.08% | 49.38% | 1.03 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 682 | 329 | 353 | 48.24% | 46.25% | 49.58% | 1.76 pp | -24 | 41 | -0.59 |
| BTC Daily | nn | NN | 682 | 319 | 363 | 46.77% | 43.33% | 48.96% | 3.23 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 682 | 297 | 385 | 43.55% | 38.33% | 42.50% | 6.45 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 682 | 292 | 390 | 42.82% | 40.42% | 43.12% | 7.18 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 692 | 274 | 418 | 39.60% | 34.58% | 39.38% | 10.40 pp | -144 | 41 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 454 | 220 | 234 | 48.46% | 44.58% | 48.46% | 1.54 pp | -14 | 44 | -0.32 |
| BTC Market Hours | nn | NN | 454 | 214 | 240 | 47.14% | 48.75% | 47.14% | 2.86 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 454 | 209 | 245 | 46.04% | 39.58% | 46.04% | 3.96 pp | -36 | 44 | -0.82 |
| BTC Market Hours | rf | RandomForest | 454 | 196 | 258 | 43.17% | 42.92% | 43.17% | 6.83 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 454 | 193 | 261 | 42.51% | 40.42% | 42.51% | 7.49 pp | -68 | 44 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 454 | 183 | 271 | 40.31% | 38.75% | 40.31% | 9.69 pp | -88 | 44 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 508 | 233 | 275 | 45.87% | 43.33% | 46.88% | 4.13 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 508 | 232 | 276 | 45.67% | 45.42% | 46.25% | 4.33 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 508 | 231 | 277 | 45.47% | 46.67% | 45.83% | 4.53 pp | -46 | 44 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 508 | 211 | 297 | 41.54% | 41.25% | 41.46% | 8.46 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 508 | 203 | 305 | 39.96% | 37.50% | 40.83% | 10.04 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 508 | 199 | 309 | 39.17% | 35.42% | 38.54% | 10.83 pp | -110 | 44 | -2.50 |

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
