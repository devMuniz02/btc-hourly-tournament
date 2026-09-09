# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T09:22:08.135363+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1306 | 1018 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1182 | 817 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 915 | 579 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 917 | 633 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 221 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 579 | 280 | 299 | 48.36% | 46.25% | 47.29% | 1.64 pp | -19 | 54 | -0.35 |
| BTC Market Hours | nn | NN | 579 | 278 | 301 | 48.01% | 52.08% | 49.79% | 1.99 pp | -23 | 54 | -0.43 |
| BTC Market Hours | transformer | Transformer | 579 | 275 | 304 | 47.50% | 48.33% | 47.50% | 2.50 pp | -29 | 54 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 807 | 389 | 418 | 48.20% | 46.67% | 47.08% | 1.80 pp | -29 | 46 | -0.63 |
| BTC Market Hours Daily | transformer | Transformer | 633 | 298 | 335 | 47.08% | 50.00% | 47.71% | 2.92 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | nn | NN | 633 | 297 | 336 | 46.92% | 47.92% | 47.92% | 3.08 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 633 | 296 | 337 | 46.76% | 48.75% | 46.88% | 3.24 pp | -41 | 54 | -0.76 |
| Consolidated Hourly | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 984 | 469 | 515 | 47.66% | 50.83% | 46.67% | 2.34 pp | -46 | 51 | -0.90 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 807 | 375 | 432 | 46.47% | 44.17% | 45.21% | 3.53 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 807 | 374 | 433 | 46.34% | 39.58% | 46.25% | 3.66 pp | -59 | 46 | -1.28 |
| BTC Hourly | transformer | Transformer | 984 | 455 | 529 | 46.24% | 43.75% | 43.54% | 3.76 pp | -74 | 51 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours | lstm | LSTM | 579 | 249 | 330 | 43.01% | 41.67% | 43.12% | 6.99 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 579 | 249 | 330 | 43.01% | 45.83% | 43.33% | 6.99 pp | -81 | 54 | -1.50 |
| BTC Market Hours | rf | RandomForest | 579 | 248 | 331 | 42.83% | 44.17% | 42.92% | 7.17 pp | -83 | 54 | -1.54 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 633 | 263 | 370 | 41.55% | 42.92% | 40.42% | 8.45 pp | -107 | 54 | -1.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 633 | 261 | 372 | 41.23% | 44.58% | 40.42% | 8.77 pp | -111 | 54 | -2.06 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 633 | 258 | 375 | 40.76% | 41.25% | 40.00% | 9.24 pp | -117 | 54 | -2.17 |
| Consolidated Hourly | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 984 | 434 | 550 | 44.11% | 41.67% | 42.71% | 5.89 pp | -116 | 51 | -2.27 |
| BTC Hourly | nn | NN | 984 | 433 | 551 | 44.00% | 41.67% | 42.08% | 6.00 pp | -118 | 51 | -2.31 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 807 | 340 | 467 | 42.13% | 35.00% | 40.42% | 7.87 pp | -127 | 46 | -2.76 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 984 | 418 | 566 | 42.48% | 37.50% | 40.62% | 7.52 pp | -148 | 51 | -2.90 |
| BTC Daily | rf | RandomForest | 807 | 335 | 472 | 41.51% | 36.67% | 41.25% | 8.49 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 984 | 405 | 579 | 41.16% | 35.83% | 38.75% | 8.84 pp | -174 | 51 | -3.41 |
| BTC Daily | xgb | XGBoost | 817 | 318 | 499 | 38.92% | 35.00% | 35.42% | 11.08 pp | -181 | 46 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 984 | 469 | 515 | 47.66% | 50.83% | 46.67% | 2.34 pp | -46 | 51 | -0.90 |
| BTC Hourly | transformer | Transformer | 984 | 455 | 529 | 46.24% | 43.75% | 43.54% | 3.76 pp | -74 | 51 | -1.45 |
| BTC Hourly | rf | RandomForest | 984 | 434 | 550 | 44.11% | 41.67% | 42.71% | 5.89 pp | -116 | 51 | -2.27 |
| BTC Hourly | nn | NN | 984 | 433 | 551 | 44.00% | 41.67% | 42.08% | 6.00 pp | -118 | 51 | -2.31 |
| BTC Hourly | lstm | LSTM | 984 | 418 | 566 | 42.48% | 37.50% | 40.62% | 7.52 pp | -148 | 51 | -2.90 |
| BTC Hourly | xgb | XGBoost | 984 | 405 | 579 | 41.16% | 35.83% | 38.75% | 8.84 pp | -174 | 51 | -3.41 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 807 | 389 | 418 | 48.20% | 46.67% | 47.08% | 1.80 pp | -29 | 46 | -0.63 |
| BTC Daily | nn | NN | 807 | 375 | 432 | 46.47% | 44.17% | 45.21% | 3.53 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 807 | 374 | 433 | 46.34% | 39.58% | 46.25% | 3.66 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 807 | 340 | 467 | 42.13% | 35.00% | 40.42% | 7.87 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 807 | 335 | 472 | 41.51% | 36.67% | 41.25% | 8.49 pp | -137 | 46 | -2.98 |
| BTC Daily | xgb | XGBoost | 817 | 318 | 499 | 38.92% | 35.00% | 35.42% | 11.08 pp | -181 | 46 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 579 | 280 | 299 | 48.36% | 46.25% | 47.29% | 1.64 pp | -19 | 54 | -0.35 |
| BTC Market Hours | nn | NN | 579 | 278 | 301 | 48.01% | 52.08% | 49.79% | 1.99 pp | -23 | 54 | -0.43 |
| BTC Market Hours | transformer | Transformer | 579 | 275 | 304 | 47.50% | 48.33% | 47.50% | 2.50 pp | -29 | 54 | -0.54 |
| BTC Market Hours | lstm | LSTM | 579 | 249 | 330 | 43.01% | 41.67% | 43.12% | 6.99 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 579 | 249 | 330 | 43.01% | 45.83% | 43.33% | 6.99 pp | -81 | 54 | -1.50 |
| BTC Market Hours | rf | RandomForest | 579 | 248 | 331 | 42.83% | 44.17% | 42.92% | 7.17 pp | -83 | 54 | -1.54 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 633 | 298 | 335 | 47.08% | 50.00% | 47.71% | 2.92 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | nn | NN | 633 | 297 | 336 | 46.92% | 47.92% | 47.92% | 3.08 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 633 | 296 | 337 | 46.76% | 48.75% | 46.88% | 3.24 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 633 | 263 | 370 | 41.55% | 42.92% | 40.42% | 8.45 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 633 | 261 | 372 | 41.23% | 44.58% | 40.42% | 8.77 pp | -111 | 54 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 633 | 258 | 375 | 40.76% | 41.25% | 40.00% | 9.24 pp | -117 | 54 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 100 | 121 | 45.25% | 45.25% | 45.25% | 4.75 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 95 | 126 | 42.99% | 42.99% | 42.99% | 7.01 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 14 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
