# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T07:21:36.842334+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1240 | 952 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1116 | 751 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 797 | 513 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 799 | 567 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 159 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 13:00:00+00:00 | 159 | 42 | 117 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 513 | 249 | 264 | 48.54% | 45.83% | 48.54% | 1.46 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 513 | 247 | 266 | 48.15% | 47.50% | 48.54% | 1.85 pp | -19 | 49 | -0.39 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 741 | 359 | 382 | 48.45% | 47.08% | 48.33% | 1.55 pp | -23 | 44 | -0.52 |
| BTC Market Hours Daily | transformer | Transformer | 567 | 270 | 297 | 47.62% | 51.67% | 48.75% | 2.38 pp | -27 | 49 | -0.55 |
| BTC Market Hours | nn | NN | 513 | 242 | 271 | 47.17% | 50.00% | 48.33% | 2.83 pp | -29 | 49 | -0.59 |
| BTC Daily | transformer | Transformer | 741 | 353 | 388 | 47.64% | 46.25% | 49.38% | 2.36 pp | -35 | 44 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 918 | 439 | 479 | 47.82% | 49.58% | 47.50% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | nn | NN | 567 | 262 | 305 | 46.21% | 45.42% | 47.50% | 3.79 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 567 | 261 | 306 | 46.03% | 48.75% | 46.25% | 3.97 pp | -45 | 49 | -0.92 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 918 | 434 | 484 | 47.28% | 47.50% | 46.25% | 2.72 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Daily | nn | NN | 741 | 342 | 399 | 46.15% | 43.33% | 46.67% | 3.85 pp | -57 | 44 | -1.30 |
| BTC Market Hours | lstm | LSTM | 513 | 223 | 290 | 43.47% | 42.50% | 43.75% | 6.53 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 513 | 222 | 291 | 43.27% | 45.00% | 43.75% | 6.73 pp | -69 | 49 | -1.41 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 513 | 211 | 302 | 41.13% | 42.92% | 41.88% | 8.87 pp | -91 | 49 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 567 | 236 | 331 | 41.62% | 42.50% | 40.83% | 8.38 pp | -95 | 49 | -1.94 |
| BTC Hourly | rf | RandomForest | 918 | 409 | 509 | 44.55% | 44.17% | 44.38% | 5.45 pp | -100 | 48 | -2.08 |
| BTC Hourly | nn | NN | 918 | 408 | 510 | 44.44% | 43.33% | 42.29% | 5.56 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 567 | 230 | 337 | 40.56% | 39.58% | 40.62% | 9.44 pp | -107 | 49 | -2.18 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 567 | 227 | 340 | 40.04% | 40.83% | 39.17% | 9.96 pp | -113 | 49 | -2.31 |
| BTC Daily | lstm | LSTM | 741 | 318 | 423 | 42.91% | 37.08% | 41.04% | 7.09 pp | -105 | 44 | -2.39 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 741 | 313 | 428 | 42.24% | 39.58% | 42.50% | 7.76 pp | -115 | 44 | -2.61 |
| BTC Hourly | lstm | LSTM | 918 | 393 | 525 | 42.81% | 39.17% | 41.67% | 7.19 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 918 | 383 | 535 | 41.72% | 39.17% | 40.00% | 8.28 pp | -152 | 48 | -3.17 |
| BTC Daily | xgb | XGBoost | 751 | 297 | 454 | 39.55% | 36.25% | 37.71% | 10.45 pp | -157 | 44 | -3.57 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 918 | 439 | 479 | 47.82% | 49.58% | 47.50% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Hourly | transformer | Transformer | 918 | 434 | 484 | 47.28% | 47.50% | 46.25% | 2.72 pp | -50 | 48 | -1.04 |
| BTC Hourly | rf | RandomForest | 918 | 409 | 509 | 44.55% | 44.17% | 44.38% | 5.45 pp | -100 | 48 | -2.08 |
| BTC Hourly | nn | NN | 918 | 408 | 510 | 44.44% | 43.33% | 42.29% | 5.56 pp | -102 | 48 | -2.12 |
| BTC Hourly | lstm | LSTM | 918 | 393 | 525 | 42.81% | 39.17% | 41.67% | 7.19 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 918 | 383 | 535 | 41.72% | 39.17% | 40.00% | 8.28 pp | -152 | 48 | -3.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 741 | 359 | 382 | 48.45% | 47.08% | 48.33% | 1.55 pp | -23 | 44 | -0.52 |
| BTC Daily | transformer | Transformer | 741 | 353 | 388 | 47.64% | 46.25% | 49.38% | 2.36 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 741 | 342 | 399 | 46.15% | 43.33% | 46.67% | 3.85 pp | -57 | 44 | -1.30 |
| BTC Daily | lstm | LSTM | 741 | 318 | 423 | 42.91% | 37.08% | 41.04% | 7.09 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 741 | 313 | 428 | 42.24% | 39.58% | 42.50% | 7.76 pp | -115 | 44 | -2.61 |
| BTC Daily | xgb | XGBoost | 751 | 297 | 454 | 39.55% | 36.25% | 37.71% | 10.45 pp | -157 | 44 | -3.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 513 | 249 | 264 | 48.54% | 45.83% | 48.54% | 1.46 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 513 | 247 | 266 | 48.15% | 47.50% | 48.54% | 1.85 pp | -19 | 49 | -0.39 |
| BTC Market Hours | nn | NN | 513 | 242 | 271 | 47.17% | 50.00% | 48.33% | 2.83 pp | -29 | 49 | -0.59 |
| BTC Market Hours | lstm | LSTM | 513 | 223 | 290 | 43.47% | 42.50% | 43.75% | 6.53 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 513 | 222 | 291 | 43.27% | 45.00% | 43.75% | 6.73 pp | -69 | 49 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 513 | 211 | 302 | 41.13% | 42.92% | 41.88% | 8.87 pp | -91 | 49 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 567 | 270 | 297 | 47.62% | 51.67% | 48.75% | 2.38 pp | -27 | 49 | -0.55 |
| BTC Market Hours Daily | nn | NN | 567 | 262 | 305 | 46.21% | 45.42% | 47.50% | 3.79 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 567 | 261 | 306 | 46.03% | 48.75% | 46.25% | 3.97 pp | -45 | 49 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 567 | 236 | 331 | 41.62% | 42.50% | 40.83% | 8.38 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 567 | 230 | 337 | 40.56% | 39.58% | 40.62% | 9.44 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | xgb | XGBoost | 567 | 227 | 340 | 40.04% | 40.83% | 39.17% | 9.96 pp | -113 | 49 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 79 | 80 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 69 | 90 | 43.40% | 43.40% | 43.40% | 6.60 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 66 | 93 | 41.51% | 41.51% | 41.51% | 8.49 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
