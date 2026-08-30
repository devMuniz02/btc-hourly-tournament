# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T00:01:15.617121+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 613 | 408 | 204 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 23:00:00+00:00 | 615 | 462 | 151 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T04:00:00+00:00 | 64 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T04:00:00+00:00 | 64 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T04:00:00+00:00 | 64 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T04:00:00+00:00 | 65 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 408 | 202 | 206 | 49.51% | 48.75% | 49.51% | 0.49 pp | -4 | 41 | -0.10 |
| BTC Daily | transformer | Transformer | 636 | 312 | 324 | 49.06% | 47.92% | 50.00% | 0.94 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 636 | 309 | 327 | 48.58% | 45.42% | 49.79% | 1.42 pp | -18 | 39 | -0.46 |
| BTC Market Hours | nn | NN | 408 | 193 | 215 | 47.30% | 50.83% | 47.30% | 2.70 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 408 | 189 | 219 | 46.32% | 42.50% | 46.32% | 3.68 pp | -30 | 41 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 462 | 214 | 248 | 46.32% | 46.25% | 46.32% | 3.68 pp | -34 | 41 | -0.83 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| BTC Market Hours Daily | transformer | Transformer | 462 | 212 | 250 | 45.89% | 46.67% | 45.89% | 4.11 pp | -38 | 41 | -0.93 |
| BTC Daily | nn | NN | 636 | 299 | 337 | 47.01% | 42.92% | 48.96% | 2.99 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 462 | 211 | 251 | 45.67% | 45.42% | 45.67% | 4.33 pp | -40 | 41 | -0.98 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 813 | 384 | 429 | 47.23% | 45.42% | 46.25% | 2.77 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 813 | 382 | 431 | 46.99% | 43.75% | 47.08% | 3.01 pp | -49 | 44 | -1.11 |
| BTC Market Hours | lstm | LSTM | 408 | 180 | 228 | 44.12% | 45.42% | 44.12% | 5.88 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 408 | 175 | 233 | 42.89% | 42.08% | 42.89% | 7.11 pp | -58 | 41 | -1.41 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 813 | 367 | 446 | 45.14% | 41.25% | 45.00% | 4.86 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 636 | 282 | 354 | 44.34% | 42.50% | 43.75% | 5.66 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 813 | 364 | 449 | 44.77% | 44.58% | 44.58% | 5.23 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 408 | 164 | 244 | 40.20% | 38.75% | 40.20% | 9.80 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 462 | 190 | 272 | 41.13% | 41.67% | 41.13% | 8.87 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 462 | 189 | 273 | 40.91% | 40.00% | 40.91% | 9.09 pp | -84 | 41 | -2.05 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| BTC Hourly | lstm | LSTM | 813 | 355 | 458 | 43.67% | 42.08% | 44.38% | 6.33 pp | -103 | 44 | -2.34 |
| BTC Daily | rf | RandomForest | 636 | 272 | 364 | 42.77% | 42.08% | 43.54% | 7.23 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 462 | 181 | 281 | 39.18% | 36.25% | 39.18% | 10.82 pp | -100 | 41 | -2.44 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 462 | 214 | 248 | 46.32% | 46.25% | 46.32% | 3.68 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 462 | 212 | 250 | 45.89% | 46.67% | 45.89% | 4.11 pp | -38 | 41 | -0.93 |
| BTC Market Hours Daily | nn | NN | 462 | 211 | 251 | 45.67% | 45.42% | 45.67% | 4.33 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 462 | 190 | 272 | 41.13% | 41.67% | 41.13% | 8.87 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 462 | 189 | 273 | 40.91% | 40.00% | 40.91% | 9.09 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 462 | 181 | 281 | 39.18% | 36.25% | 39.18% | 10.82 pp | -100 | 41 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
