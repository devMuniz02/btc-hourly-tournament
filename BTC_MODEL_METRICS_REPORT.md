# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T06:32:53.469843+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1304 | 1016 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1180 | 815 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 913 | 577 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 915 | 631 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 220 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 577 | 280 | 297 | 48.53% | 47.08% | 47.71% | 1.47 pp | -17 | 54 | -0.31 |
| BTC Market Hours | nn | NN | 577 | 276 | 301 | 47.83% | 52.08% | 49.38% | 2.17 pp | -25 | 54 | -0.46 |
| BTC Market Hours | transformer | Transformer | 577 | 273 | 304 | 47.31% | 47.92% | 47.08% | 2.69 pp | -31 | 54 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 805 | 389 | 416 | 48.32% | 46.67% | 47.50% | 1.68 pp | -27 | 46 | -0.59 |
| BTC Market Hours Daily | transformer | Transformer | 631 | 296 | 335 | 46.91% | 49.17% | 47.50% | 3.09 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 631 | 295 | 336 | 46.75% | 47.08% | 47.92% | 3.25 pp | -41 | 54 | -0.76 |
| Consolidated Hourly | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 631 | 294 | 337 | 46.59% | 48.33% | 46.67% | 3.41 pp | -43 | 54 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 982 | 469 | 513 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 805 | 374 | 431 | 46.46% | 44.17% | 45.00% | 3.54 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 805 | 373 | 432 | 46.34% | 39.17% | 46.04% | 3.66 pp | -59 | 46 | -1.28 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 982 | 455 | 527 | 46.33% | 43.75% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 577 | 249 | 328 | 43.15% | 42.50% | 43.33% | 6.85 pp | -79 | 54 | -1.46 |
| Consolidated Hourly | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 577 | 248 | 329 | 42.98% | 45.00% | 43.33% | 7.02 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 577 | 248 | 329 | 42.98% | 46.25% | 43.12% | 7.02 pp | -81 | 54 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 631 | 263 | 368 | 41.68% | 43.33% | 40.62% | 8.32 pp | -105 | 54 | -1.94 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 631 | 261 | 370 | 41.36% | 45.00% | 40.62% | 8.64 pp | -109 | 54 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 631 | 258 | 373 | 40.89% | 41.67% | 40.21% | 9.11 pp | -115 | 54 | -2.13 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 982 | 434 | 548 | 44.20% | 41.67% | 42.71% | 5.80 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 982 | 433 | 549 | 44.09% | 41.67% | 42.08% | 5.91 pp | -116 | 51 | -2.27 |
| BTC Daily | lstm | LSTM | 805 | 340 | 465 | 42.24% | 35.42% | 40.62% | 7.76 pp | -125 | 46 | -2.72 |
| BTC Hourly | lstm | LSTM | 982 | 418 | 564 | 42.57% | 37.92% | 40.83% | 7.43 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 805 | 334 | 471 | 41.49% | 36.67% | 41.04% | 8.51 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 982 | 405 | 577 | 41.24% | 35.83% | 38.96% | 8.76 pp | -172 | 51 | -3.37 |
| BTC Daily | xgb | XGBoost | 815 | 317 | 498 | 38.90% | 35.00% | 35.62% | 11.10 pp | -181 | 46 | -3.93 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 982 | 469 | 513 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| BTC Hourly | transformer | Transformer | 982 | 455 | 527 | 46.33% | 43.75% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Hourly | rf | RandomForest | 982 | 434 | 548 | 44.20% | 41.67% | 42.71% | 5.80 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 982 | 433 | 549 | 44.09% | 41.67% | 42.08% | 5.91 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 982 | 418 | 564 | 42.57% | 37.92% | 40.83% | 7.43 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 982 | 405 | 577 | 41.24% | 35.83% | 38.96% | 8.76 pp | -172 | 51 | -3.37 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 805 | 389 | 416 | 48.32% | 46.67% | 47.50% | 1.68 pp | -27 | 46 | -0.59 |
| BTC Daily | nn | NN | 805 | 374 | 431 | 46.46% | 44.17% | 45.00% | 3.54 pp | -57 | 46 | -1.24 |
| BTC Daily | transformer | Transformer | 805 | 373 | 432 | 46.34% | 39.17% | 46.04% | 3.66 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 805 | 340 | 465 | 42.24% | 35.42% | 40.62% | 7.76 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 805 | 334 | 471 | 41.49% | 36.67% | 41.04% | 8.51 pp | -137 | 46 | -2.98 |
| BTC Daily | xgb | XGBoost | 815 | 317 | 498 | 38.90% | 35.00% | 35.62% | 11.10 pp | -181 | 46 | -3.93 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 577 | 280 | 297 | 48.53% | 47.08% | 47.71% | 1.47 pp | -17 | 54 | -0.31 |
| BTC Market Hours | nn | NN | 577 | 276 | 301 | 47.83% | 52.08% | 49.38% | 2.17 pp | -25 | 54 | -0.46 |
| BTC Market Hours | transformer | Transformer | 577 | 273 | 304 | 47.31% | 47.92% | 47.08% | 2.69 pp | -31 | 54 | -0.57 |
| BTC Market Hours | lstm | LSTM | 577 | 249 | 328 | 43.15% | 42.50% | 43.33% | 6.85 pp | -79 | 54 | -1.46 |
| BTC Market Hours | rf | RandomForest | 577 | 248 | 329 | 42.98% | 45.00% | 43.33% | 7.02 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 577 | 248 | 329 | 42.98% | 46.25% | 43.12% | 7.02 pp | -81 | 54 | -1.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 631 | 296 | 335 | 46.91% | 49.17% | 47.50% | 3.09 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 631 | 295 | 336 | 46.75% | 47.08% | 47.92% | 3.25 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 631 | 294 | 337 | 46.59% | 48.33% | 46.67% | 3.41 pp | -43 | 54 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 631 | 263 | 368 | 41.68% | 43.33% | 40.62% | 8.32 pp | -105 | 54 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 631 | 261 | 370 | 41.36% | 45.00% | 40.62% | 8.64 pp | -109 | 54 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 631 | 258 | 373 | 40.89% | 41.67% | 40.21% | 9.11 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
