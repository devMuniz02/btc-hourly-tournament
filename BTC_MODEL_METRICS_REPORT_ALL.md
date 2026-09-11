# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T11:50:52.429489+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1339 | 1051 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1215 | 850 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 974 | 612 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 976 | 666 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 251 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T00:00:00+00:00 | 252 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 612 | 295 | 317 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 612 | 293 | 319 | 47.88% | 50.83% | 49.79% | 2.12 pp | -26 | 56 | -0.46 |
| BTC Market Hours Daily | nn | NN | 666 | 314 | 352 | 47.15% | 50.00% | 48.33% | 2.85 pp | -38 | 56 | -0.68 |
| BTC Market Hours | transformer | Transformer | 612 | 286 | 326 | 46.73% | 46.25% | 45.62% | 3.27 pp | -40 | 56 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 666 | 312 | 354 | 46.85% | 49.17% | 47.29% | 3.15 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 666 | 310 | 356 | 46.55% | 48.33% | 48.12% | 3.45 pp | -46 | 56 | -0.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 400 | 440 | 47.62% | 44.17% | 46.04% | 2.38 pp | -40 | 48 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1017 | 482 | 535 | 47.39% | 47.92% | 46.04% | 2.61 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.42% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| Consolidated Hourly | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 116 | 135 | 46.22% | 46.25% | 46.22% | 3.78 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1017 | 476 | 541 | 46.80% | 46.67% | 44.79% | 3.20 pp | -65 | 52 | -1.25 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.50% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.42% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Market Hours | lstm | LSTM | 612 | 262 | 350 | 42.81% | 42.92% | 42.92% | 7.19 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 612 | 261 | 351 | 42.65% | 42.92% | 41.88% | 7.35 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 612 | 260 | 352 | 42.48% | 46.25% | 43.12% | 7.52 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 111 | 140 | 44.22% | 43.75% | 44.22% | 5.78 pp | -29 | 16 | -1.81 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 666 | 276 | 390 | 41.44% | 43.33% | 41.67% | 8.56 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 666 | 273 | 393 | 40.99% | 44.17% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 666 | 273 | 393 | 40.99% | 43.75% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| Consolidated Hourly | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 107 | 144 | 42.63% | 42.92% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 107 | 144 | 42.63% | 42.08% | 42.63% | 7.37 pp | -37 | 16 | -2.31 |
| BTC Hourly | nn | NN | 1017 | 448 | 569 | 44.05% | 41.25% | 40.83% | 5.95 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1017 | 445 | 572 | 43.76% | 40.42% | 42.08% | 6.24 pp | -127 | 52 | -2.44 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 840 | 354 | 486 | 42.14% | 35.83% | 39.58% | 7.86 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.25% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1017 | 428 | 589 | 42.08% | 35.00% | 39.17% | 7.92 pp | -161 | 52 | -3.10 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1017 | 416 | 601 | 40.90% | 34.58% | 37.50% | 9.10 pp | -185 | 52 | -3.56 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.50% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1017 | 482 | 535 | 47.39% | 47.92% | 46.04% | 2.61 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1017 | 476 | 541 | 46.80% | 46.67% | 44.79% | 3.20 pp | -65 | 52 | -1.25 |
| BTC Hourly | nn | NN | 1017 | 448 | 569 | 44.05% | 41.25% | 40.83% | 5.95 pp | -121 | 52 | -2.33 |
| BTC Hourly | rf | RandomForest | 1017 | 445 | 572 | 43.76% | 40.42% | 42.08% | 6.24 pp | -127 | 52 | -2.44 |
| BTC Hourly | lstm | LSTM | 1017 | 428 | 589 | 42.08% | 35.00% | 39.17% | 7.92 pp | -161 | 52 | -3.10 |
| BTC Hourly | xgb | XGBoost | 1017 | 416 | 601 | 40.90% | 34.58% | 37.50% | 9.10 pp | -185 | 52 | -3.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 840 | 400 | 440 | 47.62% | 44.17% | 46.04% | 2.38 pp | -40 | 48 | -0.83 |
| BTC Daily | nn | NN | 840 | 392 | 448 | 46.67% | 45.42% | 45.21% | 3.33 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 840 | 387 | 453 | 46.07% | 37.50% | 44.38% | 3.93 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 840 | 354 | 486 | 42.14% | 35.83% | 39.58% | 7.86 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 840 | 348 | 492 | 41.43% | 36.25% | 40.83% | 8.57 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 850 | 336 | 514 | 39.53% | 37.50% | 36.88% | 10.47 pp | -178 | 48 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 612 | 295 | 317 | 48.20% | 46.67% | 47.71% | 1.80 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 612 | 293 | 319 | 47.88% | 50.83% | 49.79% | 2.12 pp | -26 | 56 | -0.46 |
| BTC Market Hours | transformer | Transformer | 612 | 286 | 326 | 46.73% | 46.25% | 45.62% | 3.27 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 612 | 262 | 350 | 42.81% | 42.92% | 42.92% | 7.19 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 612 | 261 | 351 | 42.65% | 42.92% | 41.88% | 7.35 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 612 | 260 | 352 | 42.48% | 46.25% | 43.12% | 7.52 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 666 | 314 | 352 | 47.15% | 50.00% | 48.33% | 2.85 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 666 | 312 | 354 | 46.85% | 49.17% | 47.29% | 3.15 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 666 | 310 | 356 | 46.55% | 48.33% | 48.12% | 3.45 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 666 | 276 | 390 | 41.44% | 43.33% | 41.67% | 8.56 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 666 | 273 | 393 | 40.99% | 44.17% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 666 | 273 | 393 | 40.99% | 43.75% | 41.04% | 9.01 pp | -120 | 56 | -2.14 |

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
