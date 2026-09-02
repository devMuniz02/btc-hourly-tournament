# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T12:05:00.117703+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1195 | 907 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1070 | 705 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 712 | 467 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 714 | 521 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 117 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 117 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 19 | 98 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 16:00:00+00:00 | 117 | 19 | 98 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 467 | 227 | 240 | 48.61% | 44.58% | 48.61% | 1.39 pp | -13 | 45 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 695 | 339 | 356 | 48.78% | 45.83% | 49.17% | 1.22 pp | -17 | 42 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 467 | 219 | 248 | 46.90% | 47.92% | 46.90% | 3.10 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 467 | 218 | 249 | 46.68% | 41.25% | 46.68% | 3.32 pp | -31 | 45 | -0.69 |
| BTC Daily | transformer | Transformer | 695 | 333 | 362 | 47.91% | 45.83% | 49.17% | 2.09 pp | -29 | 42 | -0.69 |
| Consolidated Hourly | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 521 | 239 | 282 | 45.87% | 46.67% | 46.46% | 4.13 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 873 | 414 | 459 | 47.42% | 48.33% | 47.71% | 2.58 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 521 | 238 | 283 | 45.68% | 47.08% | 46.46% | 4.32 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | nn | NN | 521 | 237 | 284 | 45.49% | 42.50% | 46.04% | 4.51 pp | -47 | 45 | -1.04 |
| BTC Hourly | transformer | Transformer | 873 | 412 | 461 | 47.19% | 48.33% | 47.50% | 2.81 pp | -49 | 46 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 695 | 322 | 373 | 46.33% | 42.50% | 48.33% | 3.67 pp | -51 | 42 | -1.21 |
| Consolidated Hourly | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 467 | 201 | 266 | 43.04% | 42.92% | 43.04% | 6.96 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 467 | 200 | 267 | 42.83% | 40.83% | 42.83% | 7.17 pp | -67 | 45 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 873 | 393 | 480 | 45.02% | 46.25% | 43.96% | 4.98 pp | -87 | 46 | -1.89 |
| Consolidated Hourly | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 467 | 190 | 277 | 40.69% | 39.58% | 40.69% | 9.31 pp | -87 | 45 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 521 | 215 | 306 | 41.27% | 40.83% | 41.25% | 8.73 pp | -91 | 45 | -2.02 |
| BTC Hourly | rf | RandomForest | 873 | 389 | 484 | 44.56% | 44.58% | 43.96% | 5.44 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 695 | 303 | 392 | 43.60% | 38.75% | 42.50% | 6.40 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 521 | 208 | 313 | 39.92% | 37.50% | 40.83% | 10.08 pp | -105 | 45 | -2.33 |
| BTC Daily | rf | RandomForest | 695 | 298 | 397 | 42.88% | 40.42% | 43.33% | 7.12 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 521 | 206 | 315 | 39.54% | 37.08% | 39.17% | 10.46 pp | -109 | 45 | -2.42 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 873 | 372 | 501 | 42.61% | 38.75% | 41.67% | 7.39 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 873 | 369 | 504 | 42.27% | 41.25% | 42.92% | 7.73 pp | -135 | 46 | -2.93 |
| BTC Daily | xgb | XGBoost | 705 | 279 | 426 | 39.57% | 35.83% | 39.38% | 10.43 pp | -147 | 42 | -3.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 873 | 414 | 459 | 47.42% | 48.33% | 47.71% | 2.58 pp | -45 | 46 | -0.98 |
| BTC Hourly | transformer | Transformer | 873 | 412 | 461 | 47.19% | 48.33% | 47.50% | 2.81 pp | -49 | 46 | -1.07 |
| BTC Hourly | nn | NN | 873 | 393 | 480 | 45.02% | 46.25% | 43.96% | 4.98 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 873 | 389 | 484 | 44.56% | 44.58% | 43.96% | 5.44 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 873 | 372 | 501 | 42.61% | 38.75% | 41.67% | 7.39 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 873 | 369 | 504 | 42.27% | 41.25% | 42.92% | 7.73 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 695 | 339 | 356 | 48.78% | 45.83% | 49.17% | 1.22 pp | -17 | 42 | -0.40 |
| BTC Daily | transformer | Transformer | 695 | 333 | 362 | 47.91% | 45.83% | 49.17% | 2.09 pp | -29 | 42 | -0.69 |
| BTC Daily | nn | NN | 695 | 322 | 373 | 46.33% | 42.50% | 48.33% | 3.67 pp | -51 | 42 | -1.21 |
| BTC Daily | lstm | LSTM | 695 | 303 | 392 | 43.60% | 38.75% | 42.50% | 6.40 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 695 | 298 | 397 | 42.88% | 40.42% | 43.33% | 7.12 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 705 | 279 | 426 | 39.57% | 35.83% | 39.38% | 10.43 pp | -147 | 42 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 467 | 227 | 240 | 48.61% | 44.58% | 48.61% | 1.39 pp | -13 | 45 | -0.29 |
| BTC Market Hours | nn | NN | 467 | 219 | 248 | 46.90% | 47.92% | 46.90% | 3.10 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 467 | 218 | 249 | 46.68% | 41.25% | 46.68% | 3.32 pp | -31 | 45 | -0.69 |
| BTC Market Hours | rf | RandomForest | 467 | 201 | 266 | 43.04% | 42.92% | 43.04% | 6.96 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 467 | 200 | 267 | 42.83% | 40.83% | 42.83% | 7.17 pp | -67 | 45 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 467 | 190 | 277 | 40.69% | 39.58% | 40.69% | 9.31 pp | -87 | 45 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 521 | 239 | 282 | 45.87% | 46.67% | 46.46% | 4.13 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 521 | 238 | 283 | 45.68% | 47.08% | 46.46% | 4.32 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | nn | NN | 521 | 237 | 284 | 45.49% | 42.50% | 46.04% | 4.51 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | rf | RandomForest | 521 | 215 | 306 | 41.27% | 40.83% | 41.25% | 8.73 pp | -91 | 45 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 521 | 208 | 313 | 39.92% | 37.50% | 40.83% | 10.08 pp | -105 | 45 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 521 | 206 | 315 | 39.54% | 37.08% | 39.17% | 10.46 pp | -109 | 45 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 60 | 57 | 51.28% | 51.28% | 51.28% | 1.28 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
