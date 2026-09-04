# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T22:13:52.550122+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1234 | 946 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1109 | 744 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 21:00:00+00:00 | 787 | 506 | 280 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 21:00:00+00:00 | 789 | 560 | 227 | 2 |
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
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 506 | 244 | 262 | 48.22% | 44.58% | 48.12% | 1.78 pp | -18 | 48 | -0.38 |
| BTC Market Hours | transformer | Transformer | 506 | 242 | 264 | 47.83% | 46.25% | 48.33% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 734 | 355 | 379 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 43 | -0.56 |
| BTC Market Hours | nn | NN | 506 | 239 | 267 | 47.23% | 50.00% | 48.12% | 2.77 pp | -28 | 48 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 560 | 265 | 295 | 47.32% | 50.42% | 48.33% | 2.68 pp | -30 | 48 | -0.62 |
| BTC Daily | transformer | Transformer | 734 | 350 | 384 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 912 | 437 | 475 | 47.92% | 50.83% | 47.92% | 2.08 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | nn | NN | 560 | 260 | 300 | 46.43% | 45.83% | 47.92% | 3.57 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 560 | 259 | 301 | 46.25% | 49.17% | 46.67% | 3.75 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 912 | 431 | 481 | 47.26% | 47.92% | 46.67% | 2.74 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 734 | 340 | 394 | 46.32% | 44.58% | 47.29% | 3.68 pp | -54 | 43 | -1.26 |
| BTC Market Hours | lstm | LSTM | 506 | 219 | 287 | 43.28% | 42.08% | 43.33% | 6.72 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 506 | 217 | 289 | 42.89% | 43.75% | 43.12% | 7.11 pp | -72 | 48 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 154 | 67 | 87 | 43.51% | 43.51% | 43.51% | 6.49 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 506 | 208 | 298 | 41.11% | 42.08% | 41.67% | 8.89 pp | -90 | 48 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 560 | 231 | 329 | 41.25% | 41.67% | 40.42% | 8.75 pp | -98 | 48 | -2.04 |
| BTC Hourly | nn | NN | 912 | 405 | 507 | 44.41% | 43.75% | 42.08% | 5.59 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 912 | 405 | 507 | 44.41% | 44.17% | 43.96% | 5.59 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 560 | 226 | 334 | 40.36% | 38.75% | 40.42% | 9.64 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 560 | 224 | 336 | 40.00% | 40.83% | 39.17% | 10.00 pp | -112 | 48 | -2.33 |
| BTC Daily | lstm | LSTM | 734 | 315 | 419 | 42.92% | 36.67% | 41.25% | 7.08 pp | -104 | 43 | -2.42 |
| Consolidated Hourly | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 11 | -2.55 |
| BTC Daily | rf | RandomForest | 734 | 311 | 423 | 42.37% | 40.00% | 43.12% | 7.63 pp | -112 | 43 | -2.60 |
| BTC Hourly | lstm | LSTM | 912 | 390 | 522 | 42.76% | 39.58% | 41.46% | 7.24 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 912 | 382 | 530 | 41.89% | 40.42% | 40.62% | 8.11 pp | -148 | 48 | -3.08 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 3 | -3.67 |
| BTC Daily | xgb | XGBoost | 744 | 293 | 451 | 39.38% | 35.83% | 37.92% | 10.62 pp | -158 | 43 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 912 | 437 | 475 | 47.92% | 50.83% | 47.92% | 2.08 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 912 | 431 | 481 | 47.26% | 47.92% | 46.67% | 2.74 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 912 | 405 | 507 | 44.41% | 43.75% | 42.08% | 5.59 pp | -102 | 48 | -2.12 |
| BTC Hourly | rf | RandomForest | 912 | 405 | 507 | 44.41% | 44.17% | 43.96% | 5.59 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 912 | 390 | 522 | 42.76% | 39.58% | 41.46% | 7.24 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 912 | 382 | 530 | 41.89% | 40.42% | 40.62% | 8.11 pp | -148 | 48 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 734 | 355 | 379 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 734 | 350 | 384 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 734 | 340 | 394 | 46.32% | 44.58% | 47.29% | 3.68 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 734 | 315 | 419 | 42.92% | 36.67% | 41.25% | 7.08 pp | -104 | 43 | -2.42 |
| BTC Daily | rf | RandomForest | 734 | 311 | 423 | 42.37% | 40.00% | 43.12% | 7.63 pp | -112 | 43 | -2.60 |
| BTC Daily | xgb | XGBoost | 744 | 293 | 451 | 39.38% | 35.83% | 37.92% | 10.62 pp | -158 | 43 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 506 | 244 | 262 | 48.22% | 44.58% | 48.12% | 1.78 pp | -18 | 48 | -0.38 |
| BTC Market Hours | transformer | Transformer | 506 | 242 | 264 | 47.83% | 46.25% | 48.33% | 2.17 pp | -22 | 48 | -0.46 |
| BTC Market Hours | nn | NN | 506 | 239 | 267 | 47.23% | 50.00% | 48.12% | 2.77 pp | -28 | 48 | -0.58 |
| BTC Market Hours | lstm | LSTM | 506 | 219 | 287 | 43.28% | 42.08% | 43.33% | 6.72 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 506 | 217 | 289 | 42.89% | 43.75% | 43.12% | 7.11 pp | -72 | 48 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 506 | 208 | 298 | 41.11% | 42.08% | 41.67% | 8.89 pp | -90 | 48 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 560 | 265 | 295 | 47.32% | 50.42% | 48.33% | 2.68 pp | -30 | 48 | -0.62 |
| BTC Market Hours Daily | nn | NN | 560 | 260 | 300 | 46.43% | 45.83% | 47.92% | 3.57 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 560 | 259 | 301 | 46.25% | 49.17% | 46.67% | 3.75 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 560 | 231 | 329 | 41.25% | 41.67% | 40.42% | 8.75 pp | -98 | 48 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 560 | 226 | 334 | 40.36% | 38.75% | 40.42% | 9.64 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 560 | 224 | 336 | 40.00% | 40.83% | 39.17% | 10.00 pp | -112 | 48 | -2.33 |

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
