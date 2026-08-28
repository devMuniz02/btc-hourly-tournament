# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T02:01:31.965168+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 811 | 308 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 975 | 610 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 552 | 372 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 554 | 426 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 1 | 34 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 1 | 34 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| BTC Daily | transformer | Transformer | 600 | 297 | 303 | 49.50% | 51.25% | 50.21% | 0.50 pp | -6 | 38 | -0.16 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 372 | 183 | 189 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 600 | 294 | 306 | 49.00% | 47.92% | 49.79% | 1.00 pp | -12 | 38 | -0.32 |
| BTC Market Hours | transformer | Transformer | 372 | 175 | 197 | 47.04% | 45.00% | 47.04% | 2.96 pp | -22 | 38 | -0.58 |
| Consolidated Hourly | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 372 | 171 | 201 | 45.97% | 48.75% | 45.97% | 4.03 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 600 | 283 | 317 | 47.17% | 45.00% | 48.54% | 2.83 pp | -34 | 38 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 426 | 194 | 232 | 45.54% | 45.42% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 426 | 194 | 232 | 45.54% | 46.67% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 426 | 194 | 232 | 45.54% | 47.92% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 777 | 367 | 410 | 47.23% | 44.17% | 47.29% | 2.77 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 777 | 364 | 413 | 46.85% | 42.92% | 46.04% | 3.15 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 372 | 159 | 213 | 42.74% | 42.92% | 42.74% | 7.26 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 372 | 158 | 214 | 42.47% | 40.83% | 42.47% | 7.53 pp | -56 | 38 | -1.47 |
| BTC Daily | lstm | LSTM | 600 | 268 | 332 | 44.67% | 43.33% | 45.00% | 5.33 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 777 | 349 | 428 | 44.92% | 40.42% | 45.83% | 5.08 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 777 | 348 | 429 | 44.79% | 43.75% | 44.58% | 5.21 pp | -81 | 42 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 372 | 149 | 223 | 40.05% | 40.00% | 40.05% | 9.95 pp | -74 | 38 | -1.95 |
| BTC Daily | rf | RandomForest | 600 | 261 | 339 | 43.50% | 45.42% | 44.38% | 6.50 pp | -78 | 38 | -2.05 |
| BTC Hourly | lstm | LSTM | 777 | 344 | 433 | 44.27% | 43.33% | 46.04% | 5.73 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | rf | RandomForest | 426 | 172 | 254 | 40.38% | 40.00% | 40.38% | 9.62 pp | -82 | 38 | -2.16 |
| Consolidated Hourly | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 426 | 168 | 258 | 39.44% | 38.33% | 39.44% | 10.56 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 426 | 167 | 259 | 39.20% | 37.92% | 39.20% | 10.80 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 777 | 333 | 444 | 42.86% | 40.42% | 44.17% | 7.14 pp | -111 | 42 | -2.64 |
| BTC Daily | xgb | XGBoost | 610 | 246 | 364 | 40.33% | 36.25% | 40.42% | 9.67 pp | -118 | 38 | -3.11 |
| Consolidated Hourly | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 777 | 367 | 410 | 47.23% | 44.17% | 47.29% | 2.77 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 777 | 364 | 413 | 46.85% | 42.92% | 46.04% | 3.15 pp | -49 | 42 | -1.17 |
| BTC Hourly | nn | NN | 777 | 349 | 428 | 44.92% | 40.42% | 45.83% | 5.08 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 777 | 348 | 429 | 44.79% | 43.75% | 44.58% | 5.21 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 777 | 344 | 433 | 44.27% | 43.33% | 46.04% | 5.73 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 777 | 333 | 444 | 42.86% | 40.42% | 44.17% | 7.14 pp | -111 | 42 | -2.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 600 | 297 | 303 | 49.50% | 51.25% | 50.21% | 0.50 pp | -6 | 38 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 600 | 294 | 306 | 49.00% | 47.92% | 49.79% | 1.00 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 600 | 283 | 317 | 47.17% | 45.00% | 48.54% | 2.83 pp | -34 | 38 | -0.89 |
| BTC Daily | lstm | LSTM | 600 | 268 | 332 | 44.67% | 43.33% | 45.00% | 5.33 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 600 | 261 | 339 | 43.50% | 45.42% | 44.38% | 6.50 pp | -78 | 38 | -2.05 |
| BTC Daily | xgb | XGBoost | 610 | 246 | 364 | 40.33% | 36.25% | 40.42% | 9.67 pp | -118 | 38 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 372 | 183 | 189 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 372 | 175 | 197 | 47.04% | 45.00% | 47.04% | 2.96 pp | -22 | 38 | -0.58 |
| BTC Market Hours | nn | NN | 372 | 171 | 201 | 45.97% | 48.75% | 45.97% | 4.03 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 372 | 159 | 213 | 42.74% | 42.92% | 42.74% | 7.26 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 372 | 158 | 214 | 42.47% | 40.83% | 42.47% | 7.53 pp | -56 | 38 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 372 | 149 | 223 | 40.05% | 40.00% | 40.05% | 9.95 pp | -74 | 38 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 426 | 194 | 232 | 45.54% | 45.42% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 426 | 194 | 232 | 45.54% | 46.67% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 426 | 194 | 232 | 45.54% | 47.92% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 426 | 172 | 254 | 40.38% | 40.00% | 40.38% | 9.62 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 426 | 168 | 258 | 39.44% | 38.33% | 39.44% | 10.56 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 426 | 167 | 259 | 39.20% | 37.92% | 39.20% | 10.80 pp | -92 | 38 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Hourly | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Hourly | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
