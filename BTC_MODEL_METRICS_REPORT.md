# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T04:02:27.131658+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1189 | 901 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1065 | 700 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 707 | 462 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 709 | 516 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 113 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 462 | 225 | 237 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 690 | 338 | 352 | 48.99% | 46.25% | 49.58% | 1.01 pp | -14 | 42 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| BTC Daily | transformer | Transformer | 690 | 332 | 358 | 48.12% | 46.67% | 49.38% | 1.88 pp | -26 | 42 | -0.62 |
| BTC Market Hours | nn | NN | 462 | 217 | 245 | 46.97% | 48.33% | 46.97% | 3.03 pp | -28 | 45 | -0.62 |
| BTC Market Hours | transformer | Transformer | 462 | 214 | 248 | 46.32% | 40.42% | 46.32% | 3.68 pp | -34 | 45 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 516 | 238 | 278 | 46.12% | 47.08% | 46.46% | 3.88 pp | -40 | 45 | -0.89 |
| BTC Market Hours Daily | nn | NN | 516 | 236 | 280 | 45.74% | 43.33% | 46.46% | 4.26 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 516 | 236 | 280 | 45.74% | 47.50% | 46.25% | 4.26 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 867 | 409 | 458 | 47.17% | 46.25% | 47.50% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 690 | 322 | 368 | 46.67% | 42.92% | 49.17% | 3.33 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 867 | 407 | 460 | 46.94% | 47.50% | 46.88% | 3.06 pp | -53 | 46 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| BTC Market Hours | rf | RandomForest | 462 | 200 | 262 | 43.29% | 43.33% | 43.29% | 6.71 pp | -62 | 45 | -1.38 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 462 | 196 | 266 | 42.42% | 40.00% | 42.42% | 7.58 pp | -70 | 45 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 462 | 189 | 273 | 40.91% | 40.00% | 40.91% | 9.09 pp | -84 | 45 | -1.87 |
| BTC Hourly | nn | NN | 867 | 390 | 477 | 44.98% | 45.83% | 44.17% | 5.02 pp | -87 | 46 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 516 | 215 | 301 | 41.67% | 42.08% | 41.88% | 8.33 pp | -86 | 45 | -1.91 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 867 | 386 | 481 | 44.52% | 44.17% | 44.38% | 5.48 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 690 | 301 | 389 | 43.62% | 38.75% | 42.71% | 6.38 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 516 | 207 | 309 | 40.12% | 38.33% | 40.83% | 9.88 pp | -102 | 45 | -2.27 |
| BTC Daily | rf | RandomForest | 690 | 297 | 393 | 43.04% | 40.42% | 43.54% | 6.96 pp | -96 | 42 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 516 | 205 | 311 | 39.73% | 37.08% | 39.38% | 10.27 pp | -106 | 45 | -2.36 |
| BTC Hourly | lstm | LSTM | 867 | 369 | 498 | 42.56% | 38.33% | 42.29% | 7.44 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 867 | 365 | 502 | 42.10% | 40.42% | 43.12% | 7.90 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 700 | 278 | 422 | 39.71% | 36.25% | 39.58% | 10.29 pp | -144 | 42 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 867 | 409 | 458 | 47.17% | 46.25% | 47.50% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 867 | 407 | 460 | 46.94% | 47.50% | 46.88% | 3.06 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 867 | 390 | 477 | 44.98% | 45.83% | 44.17% | 5.02 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 867 | 386 | 481 | 44.52% | 44.17% | 44.38% | 5.48 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 867 | 369 | 498 | 42.56% | 38.33% | 42.29% | 7.44 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 867 | 365 | 502 | 42.10% | 40.42% | 43.12% | 7.90 pp | -137 | 46 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 690 | 338 | 352 | 48.99% | 46.25% | 49.58% | 1.01 pp | -14 | 42 | -0.33 |
| BTC Daily | transformer | Transformer | 690 | 332 | 358 | 48.12% | 46.67% | 49.38% | 1.88 pp | -26 | 42 | -0.62 |
| BTC Daily | nn | NN | 690 | 322 | 368 | 46.67% | 42.92% | 49.17% | 3.33 pp | -46 | 42 | -1.10 |
| BTC Daily | lstm | LSTM | 690 | 301 | 389 | 43.62% | 38.75% | 42.71% | 6.38 pp | -88 | 42 | -2.10 |
| BTC Daily | rf | RandomForest | 690 | 297 | 393 | 43.04% | 40.42% | 43.54% | 6.96 pp | -96 | 42 | -2.29 |
| BTC Daily | xgb | XGBoost | 700 | 278 | 422 | 39.71% | 36.25% | 39.58% | 10.29 pp | -144 | 42 | -3.43 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 462 | 225 | 237 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| BTC Market Hours | nn | NN | 462 | 217 | 245 | 46.97% | 48.33% | 46.97% | 3.03 pp | -28 | 45 | -0.62 |
| BTC Market Hours | transformer | Transformer | 462 | 214 | 248 | 46.32% | 40.42% | 46.32% | 3.68 pp | -34 | 45 | -0.76 |
| BTC Market Hours | rf | RandomForest | 462 | 200 | 262 | 43.29% | 43.33% | 43.29% | 6.71 pp | -62 | 45 | -1.38 |
| BTC Market Hours | lstm | LSTM | 462 | 196 | 266 | 42.42% | 40.00% | 42.42% | 7.58 pp | -70 | 45 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 462 | 189 | 273 | 40.91% | 40.00% | 40.91% | 9.09 pp | -84 | 45 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 516 | 238 | 278 | 46.12% | 47.08% | 46.46% | 3.88 pp | -40 | 45 | -0.89 |
| BTC Market Hours Daily | nn | NN | 516 | 236 | 280 | 45.74% | 43.33% | 46.46% | 4.26 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 516 | 236 | 280 | 45.74% | 47.50% | 46.25% | 4.26 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 516 | 215 | 301 | 41.67% | 42.08% | 41.88% | 8.33 pp | -86 | 45 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 516 | 207 | 309 | 40.12% | 38.33% | 40.83% | 9.88 pp | -102 | 45 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 516 | 205 | 311 | 39.73% | 37.08% | 39.38% | 10.27 pp | -106 | 45 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
