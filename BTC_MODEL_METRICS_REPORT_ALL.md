# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T08:47:35.678025+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1142 | 854 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1018 | 653 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 621 | 415 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 622 | 468 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 415 | 206 | 209 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | mlp_sklearn | MLPClassifier | 643 | 312 | 331 | 48.52% | 45.83% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 643 | 312 | 331 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 415 | 196 | 219 | 47.23% | 50.83% | 47.23% | 2.77 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 415 | 191 | 224 | 46.02% | 42.08% | 46.02% | 3.98 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 468 | 217 | 251 | 46.37% | 46.67% | 46.37% | 3.63 pp | -34 | 41 | -0.83 |
| BTC Daily | nn | NN | 643 | 302 | 341 | 46.97% | 42.92% | 49.17% | 3.03 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 468 | 214 | 254 | 45.73% | 45.42% | 45.73% | 4.27 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 820 | 388 | 432 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | nn | NN | 468 | 213 | 255 | 45.51% | 44.58% | 45.51% | 4.49 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 820 | 386 | 434 | 47.07% | 43.33% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 415 | 183 | 232 | 44.10% | 44.17% | 44.10% | 5.90 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 415 | 180 | 235 | 43.37% | 42.50% | 43.37% | 6.63 pp | -55 | 41 | -1.34 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 820 | 371 | 449 | 45.24% | 42.50% | 45.00% | 4.76 pp | -78 | 44 | -1.77 |
| BTC Daily | lstm | LSTM | 643 | 284 | 359 | 44.17% | 41.25% | 43.54% | 5.83 pp | -75 | 40 | -1.88 |
| BTC Hourly | rf | RandomForest | 820 | 367 | 453 | 44.76% | 45.00% | 44.58% | 5.24 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 468 | 193 | 275 | 41.24% | 42.08% | 41.24% | 8.76 pp | -82 | 41 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 415 | 165 | 250 | 39.76% | 37.08% | 39.76% | 10.24 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 468 | 191 | 277 | 40.81% | 40.00% | 40.81% | 9.19 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 643 | 274 | 369 | 42.61% | 41.25% | 43.54% | 7.39 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 820 | 356 | 464 | 43.41% | 41.25% | 43.75% | 6.59 pp | -108 | 44 | -2.45 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | xgb | XGBoost | 468 | 181 | 287 | 38.68% | 35.42% | 38.68% | 11.32 pp | -106 | 41 | -2.59 |
| BTC Hourly | xgb | XGBoost | 820 | 347 | 473 | 42.32% | 40.00% | 42.71% | 7.68 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 653 | 256 | 397 | 39.20% | 30.83% | 39.17% | 10.80 pp | -141 | 40 | -3.52 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 820 | 388 | 432 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 820 | 386 | 434 | 47.07% | 43.33% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 820 | 371 | 449 | 45.24% | 42.50% | 45.00% | 4.76 pp | -78 | 44 | -1.77 |
| BTC Hourly | rf | RandomForest | 820 | 367 | 453 | 44.76% | 45.00% | 44.58% | 5.24 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 820 | 356 | 464 | 43.41% | 41.25% | 43.75% | 6.59 pp | -108 | 44 | -2.45 |
| BTC Hourly | xgb | XGBoost | 820 | 347 | 473 | 42.32% | 40.00% | 42.71% | 7.68 pp | -126 | 44 | -2.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 643 | 312 | 331 | 48.52% | 45.83% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 643 | 312 | 331 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 643 | 302 | 341 | 46.97% | 42.92% | 49.17% | 3.03 pp | -39 | 40 | -0.97 |
| BTC Daily | lstm | LSTM | 643 | 284 | 359 | 44.17% | 41.25% | 43.54% | 5.83 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 643 | 274 | 369 | 42.61% | 41.25% | 43.54% | 7.39 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 653 | 256 | 397 | 39.20% | 30.83% | 39.17% | 10.80 pp | -141 | 40 | -3.52 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 415 | 206 | 209 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 415 | 196 | 219 | 47.23% | 50.83% | 47.23% | 2.77 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 415 | 191 | 224 | 46.02% | 42.08% | 46.02% | 3.98 pp | -33 | 41 | -0.80 |
| BTC Market Hours | lstm | LSTM | 415 | 183 | 232 | 44.10% | 44.17% | 44.10% | 5.90 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 415 | 180 | 235 | 43.37% | 42.50% | 43.37% | 6.63 pp | -55 | 41 | -1.34 |
| BTC Market Hours | xgb | XGBoost | 415 | 165 | 250 | 39.76% | 37.08% | 39.76% | 10.24 pp | -85 | 41 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 468 | 217 | 251 | 46.37% | 46.67% | 46.37% | 3.63 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 468 | 214 | 254 | 45.73% | 45.42% | 45.73% | 4.27 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | nn | NN | 468 | 213 | 255 | 45.51% | 44.58% | 45.51% | 4.49 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 468 | 193 | 275 | 41.24% | 42.08% | 41.24% | 8.76 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 468 | 191 | 277 | 40.81% | 40.00% | 40.81% | 9.19 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 468 | 181 | 287 | 38.68% | 35.42% | 38.68% | 11.32 pp | -106 | 41 | -2.59 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
