# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T04:50:34.141544+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1190 | 902 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1065 | 700 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 707 | 462 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 709 | 516 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 462 | 225 | 237 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 690 | 337 | 353 | 48.84% | 45.83% | 49.38% | 1.16 pp | -16 | 42 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 462 | 217 | 245 | 46.97% | 48.33% | 46.97% | 3.03 pp | -28 | 45 | -0.62 |
| BTC Daily | transformer | Transformer | 690 | 331 | 359 | 47.97% | 46.25% | 49.17% | 2.03 pp | -28 | 42 | -0.67 |
| BTC Market Hours | transformer | Transformer | 462 | 214 | 248 | 46.32% | 40.42% | 46.32% | 3.68 pp | -34 | 45 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 516 | 238 | 278 | 46.12% | 47.08% | 46.46% | 3.88 pp | -40 | 45 | -0.89 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | nn | NN | 516 | 236 | 280 | 45.74% | 43.33% | 46.46% | 4.26 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 516 | 236 | 280 | 45.74% | 47.50% | 46.25% | 4.26 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 868 | 410 | 458 | 47.24% | 46.67% | 47.50% | 2.76 pp | -48 | 46 | -1.04 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 690 | 321 | 369 | 46.52% | 42.50% | 48.96% | 3.48 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 868 | 407 | 461 | 46.89% | 47.08% | 46.88% | 3.11 pp | -54 | 46 | -1.17 |
| BTC Market Hours | rf | RandomForest | 462 | 200 | 262 | 43.29% | 43.33% | 43.29% | 6.71 pp | -62 | 45 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 462 | 196 | 266 | 42.42% | 40.00% | 42.42% | 7.58 pp | -70 | 45 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 462 | 189 | 273 | 40.91% | 40.00% | 40.91% | 9.09 pp | -84 | 45 | -1.87 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 516 | 215 | 301 | 41.67% | 42.08% | 41.88% | 8.33 pp | -86 | 45 | -1.91 |
| BTC Hourly | nn | NN | 868 | 390 | 478 | 44.93% | 45.83% | 43.96% | 5.07 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 868 | 387 | 481 | 44.59% | 44.58% | 44.38% | 5.41 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 690 | 301 | 389 | 43.62% | 38.75% | 42.71% | 6.38 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 516 | 207 | 309 | 40.12% | 38.33% | 40.83% | 9.88 pp | -102 | 45 | -2.27 |
| BTC Daily | rf | RandomForest | 690 | 296 | 394 | 42.90% | 40.00% | 43.33% | 7.10 pp | -98 | 42 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 516 | 205 | 311 | 39.73% | 37.08% | 39.38% | 10.27 pp | -106 | 45 | -2.36 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 868 | 370 | 498 | 42.63% | 38.75% | 42.29% | 7.37 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 868 | 366 | 502 | 42.17% | 40.83% | 43.12% | 7.83 pp | -136 | 46 | -2.96 |
| BTC Daily | xgb | XGBoost | 700 | 277 | 423 | 39.57% | 35.83% | 39.38% | 10.43 pp | -146 | 42 | -3.48 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 868 | 410 | 458 | 47.24% | 46.67% | 47.50% | 2.76 pp | -48 | 46 | -1.04 |
| BTC Hourly | transformer | Transformer | 868 | 407 | 461 | 46.89% | 47.08% | 46.88% | 3.11 pp | -54 | 46 | -1.17 |
| BTC Hourly | nn | NN | 868 | 390 | 478 | 44.93% | 45.83% | 43.96% | 5.07 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 868 | 387 | 481 | 44.59% | 44.58% | 44.38% | 5.41 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 868 | 370 | 498 | 42.63% | 38.75% | 42.29% | 7.37 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 868 | 366 | 502 | 42.17% | 40.83% | 43.12% | 7.83 pp | -136 | 46 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 690 | 337 | 353 | 48.84% | 45.83% | 49.38% | 1.16 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 690 | 331 | 359 | 47.97% | 46.25% | 49.17% | 2.03 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 690 | 321 | 369 | 46.52% | 42.50% | 48.96% | 3.48 pp | -48 | 42 | -1.14 |
| BTC Daily | lstm | LSTM | 690 | 301 | 389 | 43.62% | 38.75% | 42.71% | 6.38 pp | -88 | 42 | -2.10 |
| BTC Daily | rf | RandomForest | 690 | 296 | 394 | 42.90% | 40.00% | 43.33% | 7.10 pp | -98 | 42 | -2.33 |
| BTC Daily | xgb | XGBoost | 700 | 277 | 423 | 39.57% | 35.83% | 39.38% | 10.43 pp | -146 | 42 | -3.48 |

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
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
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
