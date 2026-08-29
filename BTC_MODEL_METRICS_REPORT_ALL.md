# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T23:36:05.936281+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1135 | 847 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1011 | 646 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 22:00:00+00:00 | 612 | 408 | 203 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 22:00:00+00:00 | 613 | 461 | 150 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 0 | 63 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 03:00:00+00:00 | 63 | 0 | 63 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 408 | 202 | 206 | 49.51% | 48.75% | 49.51% | 0.49 pp | -4 | 41 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| BTC Daily | transformer | Transformer | 636 | 312 | 324 | 49.06% | 47.92% | 50.00% | 0.94 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 636 | 309 | 327 | 48.58% | 45.42% | 49.79% | 1.42 pp | -18 | 39 | -0.46 |
| BTC Market Hours | nn | NN | 408 | 193 | 215 | 47.30% | 50.83% | 47.30% | 2.70 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 408 | 189 | 219 | 46.32% | 42.50% | 46.32% | 3.68 pp | -30 | 41 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 461 | 213 | 248 | 46.20% | 45.83% | 46.20% | 3.80 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 461 | 212 | 249 | 45.99% | 46.67% | 45.99% | 4.01 pp | -37 | 41 | -0.90 |
| BTC Daily | nn | NN | 636 | 299 | 337 | 47.01% | 42.92% | 48.96% | 2.99 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 461 | 210 | 251 | 45.55% | 45.00% | 45.55% | 4.45 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| BTC Hourly | transformer | Transformer | 813 | 384 | 429 | 47.23% | 45.42% | 46.25% | 2.77 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 813 | 382 | 431 | 46.99% | 43.75% | 47.08% | 3.01 pp | -49 | 44 | -1.11 |
| BTC Market Hours | lstm | LSTM | 408 | 180 | 228 | 44.12% | 45.42% | 44.12% | 5.88 pp | -48 | 41 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 408 | 175 | 233 | 42.89% | 42.08% | 42.89% | 7.11 pp | -58 | 41 | -1.41 |
| BTC Hourly | nn | NN | 813 | 367 | 446 | 45.14% | 41.25% | 45.00% | 4.86 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 636 | 282 | 354 | 44.34% | 42.50% | 43.75% | 5.66 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 813 | 364 | 449 | 44.77% | 44.58% | 44.58% | 5.23 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 408 | 164 | 244 | 40.20% | 38.75% | 40.20% | 9.80 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 461 | 189 | 272 | 41.00% | 41.25% | 41.00% | 9.00 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 41 | -2.07 |
| BTC Hourly | lstm | LSTM | 813 | 355 | 458 | 43.67% | 42.08% | 44.38% | 6.33 pp | -103 | 44 | -2.34 |
| BTC Daily | rf | RandomForest | 636 | 272 | 364 | 42.77% | 42.08% | 43.54% | 7.23 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 461 | 181 | 280 | 39.26% | 36.67% | 39.26% | 10.74 pp | -99 | 41 | -2.41 |
| Consolidated Hourly | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |
| BTC Hourly | xgb | XGBoost | 813 | 345 | 468 | 42.44% | 40.00% | 42.71% | 7.56 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 646 | 252 | 394 | 39.01% | 30.83% | 38.96% | 10.99 pp | -142 | 39 | -3.64 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 813 | 384 | 429 | 47.23% | 45.42% | 46.25% | 2.77 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 813 | 382 | 431 | 46.99% | 43.75% | 47.08% | 3.01 pp | -49 | 44 | -1.11 |
| BTC Hourly | nn | NN | 813 | 367 | 446 | 45.14% | 41.25% | 45.00% | 4.86 pp | -79 | 44 | -1.80 |
| BTC Hourly | rf | RandomForest | 813 | 364 | 449 | 44.77% | 44.58% | 44.58% | 5.23 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 813 | 355 | 458 | 43.67% | 42.08% | 44.38% | 6.33 pp | -103 | 44 | -2.34 |
| BTC Hourly | xgb | XGBoost | 813 | 345 | 468 | 42.44% | 40.00% | 42.71% | 7.56 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 636 | 312 | 324 | 49.06% | 47.92% | 50.00% | 0.94 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 636 | 309 | 327 | 48.58% | 45.42% | 49.79% | 1.42 pp | -18 | 39 | -0.46 |
| BTC Daily | nn | NN | 636 | 299 | 337 | 47.01% | 42.92% | 48.96% | 2.99 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 636 | 282 | 354 | 44.34% | 42.50% | 43.75% | 5.66 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 636 | 272 | 364 | 42.77% | 42.08% | 43.54% | 7.23 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 646 | 252 | 394 | 39.01% | 30.83% | 38.96% | 10.99 pp | -142 | 39 | -3.64 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 408 | 202 | 206 | 49.51% | 48.75% | 49.51% | 0.49 pp | -4 | 41 | -0.10 |
| BTC Market Hours | nn | NN | 408 | 193 | 215 | 47.30% | 50.83% | 47.30% | 2.70 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 408 | 189 | 219 | 46.32% | 42.50% | 46.32% | 3.68 pp | -30 | 41 | -0.73 |
| BTC Market Hours | lstm | LSTM | 408 | 180 | 228 | 44.12% | 45.42% | 44.12% | 5.88 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 408 | 175 | 233 | 42.89% | 42.08% | 42.89% | 7.11 pp | -58 | 41 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 408 | 164 | 244 | 40.20% | 38.75% | 40.20% | 9.80 pp | -80 | 41 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 461 | 213 | 248 | 46.20% | 45.83% | 46.20% | 3.80 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 461 | 212 | 249 | 45.99% | 46.67% | 45.99% | 4.01 pp | -37 | 41 | -0.90 |
| BTC Market Hours Daily | nn | NN | 461 | 210 | 251 | 45.55% | 45.00% | 45.55% | 4.45 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 461 | 189 | 272 | 41.00% | 41.25% | 41.00% | 9.00 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 461 | 181 | 280 | 39.26% | 36.67% | 39.26% | 10.74 pp | -99 | 41 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 37 | 26 | 58.73% | 58.73% | 58.73% | 8.73 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 23 | 40 | 36.51% | 36.51% | 36.51% | 13.49 pp | -17 | 7 | -2.43 |

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
