# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T14:00:36.563913+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1341 | 1053 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1216 | 851 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 977 | 613 | 363 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 979 | 667 | 310 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 613 | 295 | 318 | 48.12% | 46.67% | 47.50% | 1.88 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 613 | 294 | 319 | 47.96% | 50.83% | 49.79% | 2.04 pp | -25 | 57 | -0.44 |
| BTC Market Hours Daily | nn | NN | 667 | 314 | 353 | 47.08% | 50.00% | 48.33% | 2.92 pp | -39 | 56 | -0.70 |
| BTC Market Hours | transformer | Transformer | 613 | 286 | 327 | 46.66% | 46.25% | 45.42% | 3.34 pp | -41 | 57 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 667 | 312 | 355 | 46.78% | 48.75% | 47.29% | 3.22 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 667 | 310 | 357 | 46.48% | 48.33% | 48.12% | 3.52 pp | -47 | 56 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 841 | 399 | 442 | 47.44% | 43.75% | 45.83% | 2.56 pp | -43 | 48 | -0.90 |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1019 | 482 | 537 | 47.30% | 47.50% | 46.04% | 2.70 pp | -55 | 52 | -1.06 |
| BTC Daily | nn | NN | 841 | 393 | 448 | 46.73% | 46.25% | 45.42% | 3.27 pp | -55 | 48 | -1.15 |
| BTC Hourly | transformer | Transformer | 1019 | 478 | 541 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 52 | -1.21 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 841 | 387 | 454 | 46.02% | 37.92% | 44.17% | 3.98 pp | -67 | 48 | -1.40 |
| BTC Market Hours | lstm | LSTM | 613 | 263 | 350 | 42.90% | 43.33% | 43.12% | 7.10 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 613 | 262 | 351 | 42.74% | 43.33% | 41.88% | 7.26 pp | -89 | 57 | -1.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 613 | 261 | 352 | 42.58% | 46.67% | 43.12% | 7.42 pp | -91 | 57 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 667 | 276 | 391 | 41.38% | 42.92% | 41.67% | 8.62 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |
| BTC Hourly | nn | NN | 1019 | 448 | 571 | 43.96% | 40.83% | 40.83% | 6.04 pp | -123 | 52 | -2.37 |
| BTC Hourly | rf | RandomForest | 1019 | 446 | 573 | 43.77% | 40.42% | 42.08% | 6.23 pp | -127 | 52 | -2.44 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| BTC Daily | lstm | LSTM | 841 | 355 | 486 | 42.21% | 35.83% | 39.58% | 7.79 pp | -131 | 48 | -2.73 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| BTC Daily | rf | RandomForest | 841 | 349 | 492 | 41.50% | 37.08% | 40.83% | 8.50 pp | -143 | 48 | -2.98 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1019 | 428 | 591 | 42.00% | 34.58% | 39.17% | 8.00 pp | -163 | 52 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| BTC Hourly | xgb | XGBoost | 1019 | 417 | 602 | 40.92% | 34.58% | 37.50% | 9.08 pp | -185 | 52 | -3.56 |
| BTC Daily | xgb | XGBoost | 851 | 336 | 515 | 39.48% | 37.92% | 36.67% | 10.52 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1019 | 482 | 537 | 47.30% | 47.50% | 46.04% | 2.70 pp | -55 | 52 | -1.06 |
| BTC Hourly | transformer | Transformer | 1019 | 478 | 541 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 52 | -1.21 |
| BTC Hourly | nn | NN | 1019 | 448 | 571 | 43.96% | 40.83% | 40.83% | 6.04 pp | -123 | 52 | -2.37 |
| BTC Hourly | rf | RandomForest | 1019 | 446 | 573 | 43.77% | 40.42% | 42.08% | 6.23 pp | -127 | 52 | -2.44 |
| BTC Hourly | lstm | LSTM | 1019 | 428 | 591 | 42.00% | 34.58% | 39.17% | 8.00 pp | -163 | 52 | -3.13 |
| BTC Hourly | xgb | XGBoost | 1019 | 417 | 602 | 40.92% | 34.58% | 37.50% | 9.08 pp | -185 | 52 | -3.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 841 | 399 | 442 | 47.44% | 43.75% | 45.83% | 2.56 pp | -43 | 48 | -0.90 |
| BTC Daily | nn | NN | 841 | 393 | 448 | 46.73% | 46.25% | 45.42% | 3.27 pp | -55 | 48 | -1.15 |
| BTC Daily | transformer | Transformer | 841 | 387 | 454 | 46.02% | 37.92% | 44.17% | 3.98 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 841 | 355 | 486 | 42.21% | 35.83% | 39.58% | 7.79 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 841 | 349 | 492 | 41.50% | 37.08% | 40.83% | 8.50 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 851 | 336 | 515 | 39.48% | 37.92% | 36.67% | 10.52 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 613 | 295 | 318 | 48.12% | 46.67% | 47.50% | 1.88 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 613 | 294 | 319 | 47.96% | 50.83% | 49.79% | 2.04 pp | -25 | 57 | -0.44 |
| BTC Market Hours | transformer | Transformer | 613 | 286 | 327 | 46.66% | 46.25% | 45.42% | 3.34 pp | -41 | 57 | -0.72 |
| BTC Market Hours | lstm | LSTM | 613 | 263 | 350 | 42.90% | 43.33% | 43.12% | 7.10 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 613 | 262 | 351 | 42.74% | 43.33% | 41.88% | 7.26 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 613 | 261 | 352 | 42.58% | 46.67% | 43.12% | 7.42 pp | -91 | 57 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 667 | 314 | 353 | 47.08% | 50.00% | 48.33% | 2.92 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 667 | 312 | 355 | 46.78% | 48.75% | 47.29% | 3.22 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 667 | 310 | 357 | 46.48% | 48.33% | 48.12% | 3.52 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 667 | 276 | 391 | 41.38% | 42.92% | 41.67% | 8.62 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
