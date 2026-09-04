# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T23:49:49.043173+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1235 | 947 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1110 | 745 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 22:00:00+00:00 | 789 | 507 | 281 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 22:00:00+00:00 | 791 | 561 | 228 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 154 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 154 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 39 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 23:00:00+00:00 | 154 | 39 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 507 | 245 | 262 | 48.32% | 45.00% | 48.33% | 1.68 pp | -17 | 48 | -0.35 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| BTC Market Hours | transformer | Transformer | 507 | 243 | 264 | 47.93% | 46.25% | 48.33% | 2.07 pp | -21 | 48 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 735 | 355 | 380 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 43 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 561 | 266 | 295 | 47.42% | 50.42% | 48.54% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours | nn | NN | 507 | 239 | 268 | 47.14% | 49.58% | 48.12% | 2.86 pp | -29 | 48 | -0.60 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 913 | 437 | 476 | 47.86% | 50.42% | 47.71% | 2.14 pp | -39 | 48 | -0.81 |
| BTC Daily | transformer | Transformer | 735 | 350 | 385 | 47.62% | 46.25% | 49.58% | 2.38 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | nn | NN | 561 | 260 | 301 | 46.35% | 45.42% | 47.71% | 3.65 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 561 | 259 | 302 | 46.17% | 49.17% | 46.67% | 3.83 pp | -43 | 48 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 913 | 431 | 482 | 47.21% | 47.92% | 46.46% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 735 | 341 | 394 | 46.39% | 44.58% | 47.29% | 3.61 pp | -53 | 43 | -1.23 |
| BTC Market Hours | lstm | LSTM | 507 | 220 | 287 | 43.39% | 42.50% | 43.54% | 6.61 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 507 | 218 | 289 | 43.00% | 43.75% | 43.33% | 7.00 pp | -71 | 48 | -1.48 |
| Consolidated Hourly | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 507 | 208 | 299 | 41.03% | 42.08% | 41.67% | 8.97 pp | -91 | 48 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 561 | 232 | 329 | 41.35% | 41.67% | 40.42% | 8.65 pp | -97 | 48 | -2.02 |
| BTC Hourly | nn | NN | 913 | 405 | 508 | 44.36% | 43.33% | 42.08% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | rf | RandomForest | 913 | 405 | 508 | 44.36% | 43.75% | 43.96% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 561 | 227 | 334 | 40.46% | 38.75% | 40.62% | 9.54 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 561 | 224 | 337 | 39.93% | 40.42% | 38.96% | 10.07 pp | -113 | 48 | -2.35 |
| BTC Daily | lstm | LSTM | 735 | 315 | 420 | 42.86% | 36.67% | 41.04% | 7.14 pp | -105 | 43 | -2.44 |
| Consolidated Hourly | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| BTC Daily | rf | RandomForest | 735 | 311 | 424 | 42.31% | 40.00% | 42.92% | 7.69 pp | -113 | 43 | -2.63 |
| BTC Hourly | lstm | LSTM | 913 | 390 | 523 | 42.72% | 39.58% | 41.25% | 7.28 pp | -133 | 48 | -2.77 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 913 | 382 | 531 | 41.84% | 40.00% | 40.42% | 8.16 pp | -149 | 48 | -3.10 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| BTC Daily | xgb | XGBoost | 745 | 293 | 452 | 39.33% | 35.83% | 37.71% | 10.67 pp | -159 | 43 | -3.70 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 913 | 437 | 476 | 47.86% | 50.42% | 47.71% | 2.14 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 913 | 431 | 482 | 47.21% | 47.92% | 46.46% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 913 | 405 | 508 | 44.36% | 43.33% | 42.08% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | rf | RandomForest | 913 | 405 | 508 | 44.36% | 43.75% | 43.96% | 5.64 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 913 | 390 | 523 | 42.72% | 39.58% | 41.25% | 7.28 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 913 | 382 | 531 | 41.84% | 40.00% | 40.42% | 8.16 pp | -149 | 48 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 735 | 355 | 380 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 735 | 350 | 385 | 47.62% | 46.25% | 49.58% | 2.38 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 735 | 341 | 394 | 46.39% | 44.58% | 47.29% | 3.61 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 735 | 315 | 420 | 42.86% | 36.67% | 41.04% | 7.14 pp | -105 | 43 | -2.44 |
| BTC Daily | rf | RandomForest | 735 | 311 | 424 | 42.31% | 40.00% | 42.92% | 7.69 pp | -113 | 43 | -2.63 |
| BTC Daily | xgb | XGBoost | 745 | 293 | 452 | 39.33% | 35.83% | 37.71% | 10.67 pp | -159 | 43 | -3.70 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 507 | 245 | 262 | 48.32% | 45.00% | 48.33% | 1.68 pp | -17 | 48 | -0.35 |
| BTC Market Hours | transformer | Transformer | 507 | 243 | 264 | 47.93% | 46.25% | 48.33% | 2.07 pp | -21 | 48 | -0.44 |
| BTC Market Hours | nn | NN | 507 | 239 | 268 | 47.14% | 49.58% | 48.12% | 2.86 pp | -29 | 48 | -0.60 |
| BTC Market Hours | lstm | LSTM | 507 | 220 | 287 | 43.39% | 42.50% | 43.54% | 6.61 pp | -67 | 48 | -1.40 |
| BTC Market Hours | rf | RandomForest | 507 | 218 | 289 | 43.00% | 43.75% | 43.33% | 7.00 pp | -71 | 48 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 507 | 208 | 299 | 41.03% | 42.08% | 41.67% | 8.97 pp | -91 | 48 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 561 | 266 | 295 | 47.42% | 50.42% | 48.54% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours Daily | nn | NN | 561 | 260 | 301 | 46.35% | 45.42% | 47.71% | 3.65 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 561 | 259 | 302 | 46.17% | 49.17% | 46.67% | 3.83 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 561 | 232 | 329 | 41.35% | 41.67% | 40.42% | 8.65 pp | -97 | 48 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 561 | 227 | 334 | 40.46% | 38.75% | 40.62% | 9.54 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 561 | 224 | 337 | 39.93% | 40.42% | 38.96% | 10.07 pp | -113 | 48 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 154 | 78 | 76 | 50.65% | 50.65% | 50.65% | 0.65 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
