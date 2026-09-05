# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T11:39:44.946056+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1243 | 955 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 754 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 800 | 516 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 802 | 570 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 163 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 516 | 250 | 266 | 48.45% | 45.83% | 48.33% | 1.55 pp | -16 | 49 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 516 | 248 | 268 | 48.06% | 47.08% | 48.54% | 1.94 pp | -20 | 49 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 744 | 362 | 382 | 48.66% | 47.50% | 48.96% | 1.34 pp | -20 | 44 | -0.45 |
| BTC Market Hours Daily | transformer | Transformer | 570 | 271 | 299 | 47.54% | 51.25% | 48.75% | 2.46 pp | -28 | 49 | -0.57 |
| BTC Market Hours | nn | NN | 516 | 243 | 273 | 47.09% | 49.58% | 48.33% | 2.91 pp | -30 | 49 | -0.61 |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 744 | 355 | 389 | 47.72% | 45.83% | 49.58% | 2.28 pp | -34 | 44 | -0.77 |
| BTC Market Hours Daily | nn | NN | 570 | 264 | 306 | 46.32% | 45.83% | 47.50% | 3.68 pp | -42 | 49 | -0.86 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 921 | 439 | 482 | 47.67% | 48.75% | 47.08% | 2.33 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 570 | 263 | 307 | 46.14% | 49.58% | 46.46% | 3.86 pp | -44 | 49 | -0.90 |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 921 | 436 | 485 | 47.34% | 47.92% | 46.04% | 2.66 pp | -49 | 48 | -1.02 |
| Consolidated Hourly | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 744 | 344 | 400 | 46.24% | 42.92% | 46.88% | 3.76 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 516 | 225 | 291 | 43.60% | 42.92% | 44.17% | 6.40 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 516 | 223 | 293 | 43.22% | 45.00% | 43.75% | 6.78 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 516 | 213 | 303 | 41.28% | 42.92% | 42.08% | 8.72 pp | -90 | 49 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 570 | 237 | 333 | 41.58% | 42.50% | 40.83% | 8.42 pp | -96 | 49 | -1.96 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 921 | 410 | 511 | 44.52% | 44.17% | 43.96% | 5.48 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 921 | 409 | 512 | 44.41% | 42.92% | 42.29% | 5.59 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 570 | 232 | 338 | 40.70% | 40.00% | 40.83% | 9.30 pp | -106 | 49 | -2.16 |
| Consolidated Hourly | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 570 | 227 | 343 | 39.82% | 40.83% | 38.96% | 10.18 pp | -116 | 49 | -2.37 |
| BTC Daily | lstm | LSTM | 744 | 318 | 426 | 42.74% | 36.25% | 40.83% | 7.26 pp | -108 | 44 | -2.45 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 744 | 314 | 430 | 42.20% | 38.75% | 42.50% | 7.80 pp | -116 | 44 | -2.64 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 921 | 393 | 528 | 42.67% | 37.92% | 41.46% | 7.33 pp | -135 | 48 | -2.81 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 921 | 385 | 536 | 41.80% | 39.17% | 40.00% | 8.20 pp | -151 | 48 | -3.15 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 754 | 298 | 456 | 39.52% | 36.67% | 37.71% | 10.48 pp | -158 | 44 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 921 | 439 | 482 | 47.67% | 48.75% | 47.08% | 2.33 pp | -43 | 48 | -0.90 |
| BTC Hourly | transformer | Transformer | 921 | 436 | 485 | 47.34% | 47.92% | 46.04% | 2.66 pp | -49 | 48 | -1.02 |
| BTC Hourly | rf | RandomForest | 921 | 410 | 511 | 44.52% | 44.17% | 43.96% | 5.48 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 921 | 409 | 512 | 44.41% | 42.92% | 42.29% | 5.59 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 921 | 393 | 528 | 42.67% | 37.92% | 41.46% | 7.33 pp | -135 | 48 | -2.81 |
| BTC Hourly | xgb | XGBoost | 921 | 385 | 536 | 41.80% | 39.17% | 40.00% | 8.20 pp | -151 | 48 | -3.15 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 744 | 362 | 382 | 48.66% | 47.50% | 48.96% | 1.34 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 744 | 355 | 389 | 47.72% | 45.83% | 49.58% | 2.28 pp | -34 | 44 | -0.77 |
| BTC Daily | nn | NN | 744 | 344 | 400 | 46.24% | 42.92% | 46.88% | 3.76 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 744 | 318 | 426 | 42.74% | 36.25% | 40.83% | 7.26 pp | -108 | 44 | -2.45 |
| BTC Daily | rf | RandomForest | 744 | 314 | 430 | 42.20% | 38.75% | 42.50% | 7.80 pp | -116 | 44 | -2.64 |
| BTC Daily | xgb | XGBoost | 754 | 298 | 456 | 39.52% | 36.67% | 37.71% | 10.48 pp | -158 | 44 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 516 | 250 | 266 | 48.45% | 45.83% | 48.33% | 1.55 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 516 | 248 | 268 | 48.06% | 47.08% | 48.54% | 1.94 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 516 | 243 | 273 | 47.09% | 49.58% | 48.33% | 2.91 pp | -30 | 49 | -0.61 |
| BTC Market Hours | lstm | LSTM | 516 | 225 | 291 | 43.60% | 42.92% | 44.17% | 6.40 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 516 | 223 | 293 | 43.22% | 45.00% | 43.75% | 6.78 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 516 | 213 | 303 | 41.28% | 42.92% | 42.08% | 8.72 pp | -90 | 49 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 570 | 271 | 299 | 47.54% | 51.25% | 48.75% | 2.46 pp | -28 | 49 | -0.57 |
| BTC Market Hours Daily | nn | NN | 570 | 264 | 306 | 46.32% | 45.83% | 47.50% | 3.68 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 570 | 263 | 307 | 46.14% | 49.58% | 46.46% | 3.86 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 570 | 237 | 333 | 41.58% | 42.50% | 40.83% | 8.42 pp | -96 | 49 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 570 | 232 | 338 | 40.70% | 40.00% | 40.83% | 9.30 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 570 | 227 | 343 | 39.82% | 40.83% | 38.96% | 10.18 pp | -116 | 49 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 162 | 74 | 88 | 45.68% | 45.68% | 45.68% | 4.32 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 162 | 68 | 94 | 41.98% | 41.98% | 41.98% | 8.02 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
