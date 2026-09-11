# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T12:52:04.638098+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1340 | 1052 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1216 | 851 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 975 | 613 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 977 | 667 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 252 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 613 | 295 | 318 | 48.12% | 46.67% | 47.50% | 1.88 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 613 | 294 | 319 | 47.96% | 50.83% | 49.79% | 2.04 pp | -25 | 57 | -0.44 |
| BTC Market Hours Daily | nn | NN | 667 | 314 | 353 | 47.08% | 50.00% | 48.33% | 2.92 pp | -39 | 56 | -0.70 |
| BTC Market Hours | transformer | Transformer | 613 | 286 | 327 | 46.66% | 46.25% | 45.42% | 3.34 pp | -41 | 57 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 667 | 312 | 355 | 46.78% | 48.75% | 47.29% | 3.22 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 667 | 310 | 357 | 46.48% | 48.33% | 48.12% | 3.52 pp | -47 | 56 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 841 | 400 | 441 | 47.56% | 44.17% | 46.04% | 2.44 pp | -41 | 48 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1018 | 482 | 536 | 47.35% | 47.50% | 46.04% | 2.65 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 841 | 393 | 448 | 46.73% | 45.83% | 45.42% | 3.27 pp | -55 | 48 | -1.15 |
| Consolidated Hourly | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1018 | 477 | 541 | 46.86% | 46.67% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| BTC Daily | transformer | Transformer | 841 | 388 | 453 | 46.14% | 37.92% | 44.38% | 3.86 pp | -65 | 48 | -1.35 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 8 | -1.38 |
| BTC Market Hours | lstm | LSTM | 613 | 263 | 350 | 42.90% | 43.33% | 43.12% | 7.10 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 613 | 262 | 351 | 42.74% | 43.33% | 41.88% | 7.26 pp | -89 | 57 | -1.56 |
| Consolidated Hourly | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 613 | 261 | 352 | 42.58% | 46.67% | 43.12% | 7.42 pp | -91 | 57 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 667 | 276 | 391 | 41.38% | 42.92% | 41.67% | 8.62 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 667 | 273 | 394 | 40.93% | 43.75% | 41.04% | 9.07 pp | -121 | 56 | -2.16 |
| Consolidated Hourly | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| BTC Hourly | nn | NN | 1018 | 448 | 570 | 44.01% | 41.25% | 40.83% | 5.99 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1018 | 446 | 572 | 43.81% | 40.83% | 42.08% | 6.19 pp | -126 | 52 | -2.42 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 841 | 355 | 486 | 42.21% | 36.25% | 39.58% | 7.79 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 841 | 349 | 492 | 41.50% | 36.67% | 40.83% | 8.50 pp | -143 | 48 | -2.98 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1018 | 428 | 590 | 42.04% | 35.00% | 39.17% | 7.96 pp | -162 | 52 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1018 | 417 | 601 | 40.96% | 35.00% | 37.50% | 9.04 pp | -184 | 52 | -3.54 |
| BTC Daily | xgb | XGBoost | 851 | 337 | 514 | 39.60% | 37.92% | 36.88% | 10.40 pp | -177 | 48 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1018 | 482 | 536 | 47.35% | 47.50% | 46.04% | 2.65 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1018 | 477 | 541 | 46.86% | 46.67% | 44.79% | 3.14 pp | -64 | 52 | -1.23 |
| BTC Hourly | nn | NN | 1018 | 448 | 570 | 44.01% | 41.25% | 40.83% | 5.99 pp | -122 | 52 | -2.35 |
| BTC Hourly | rf | RandomForest | 1018 | 446 | 572 | 43.81% | 40.83% | 42.08% | 6.19 pp | -126 | 52 | -2.42 |
| BTC Hourly | lstm | LSTM | 1018 | 428 | 590 | 42.04% | 35.00% | 39.17% | 7.96 pp | -162 | 52 | -3.12 |
| BTC Hourly | xgb | XGBoost | 1018 | 417 | 601 | 40.96% | 35.00% | 37.50% | 9.04 pp | -184 | 52 | -3.54 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 841 | 400 | 441 | 47.56% | 44.17% | 46.04% | 2.44 pp | -41 | 48 | -0.85 |
| BTC Daily | nn | NN | 841 | 393 | 448 | 46.73% | 45.83% | 45.42% | 3.27 pp | -55 | 48 | -1.15 |
| BTC Daily | transformer | Transformer | 841 | 388 | 453 | 46.14% | 37.92% | 44.38% | 3.86 pp | -65 | 48 | -1.35 |
| BTC Daily | lstm | LSTM | 841 | 355 | 486 | 42.21% | 36.25% | 39.58% | 7.79 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 841 | 349 | 492 | 41.50% | 36.67% | 40.83% | 8.50 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 851 | 337 | 514 | 39.60% | 37.92% | 36.88% | 10.40 pp | -177 | 48 | -3.69 |

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
| Consolidated Hourly | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Hourly | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
