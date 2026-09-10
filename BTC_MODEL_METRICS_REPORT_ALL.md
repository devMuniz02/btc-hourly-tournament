# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T19:50:33.066227+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1329 | 1041 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1205 | 840 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 18:00:00+00:00 | 958 | 602 | 355 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 18:00:00+00:00 | 959 | 655 | 302 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 241 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 241 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 86 | 155 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 86 | 155 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 602 | 289 | 313 | 48.01% | 46.25% | 47.08% | 1.99 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 602 | 287 | 315 | 47.67% | 51.25% | 48.96% | 2.33 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 602 | 282 | 320 | 46.84% | 45.83% | 46.04% | 3.16 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | nn | NN | 655 | 307 | 348 | 46.87% | 48.33% | 47.71% | 3.13 pp | -41 | 56 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 655 | 306 | 349 | 46.72% | 48.33% | 47.08% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 655 | 306 | 349 | 46.72% | 48.33% | 48.12% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 830 | 396 | 434 | 47.71% | 45.00% | 46.04% | 2.29 pp | -38 | 47 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1007 | 477 | 530 | 47.37% | 47.92% | 45.62% | 2.63 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 830 | 386 | 444 | 46.51% | 45.00% | 45.21% | 3.49 pp | -58 | 47 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 1007 | 469 | 538 | 46.57% | 45.83% | 44.58% | 3.43 pp | -69 | 52 | -1.33 |
| BTC Daily | transformer | Transformer | 830 | 383 | 447 | 46.14% | 37.92% | 45.00% | 3.86 pp | -64 | 47 | -1.36 |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 602 | 258 | 344 | 42.86% | 42.50% | 42.92% | 7.14 pp | -86 | 56 | -1.54 |
| BTC Market Hours | rf | RandomForest | 602 | 257 | 345 | 42.69% | 42.92% | 42.29% | 7.31 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 602 | 256 | 346 | 42.52% | 45.00% | 43.33% | 7.48 pp | -90 | 56 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 655 | 271 | 384 | 41.37% | 42.08% | 41.25% | 8.63 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 655 | 268 | 387 | 40.92% | 42.50% | 40.42% | 9.08 pp | -119 | 56 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 655 | 268 | 387 | 40.92% | 42.50% | 40.62% | 9.08 pp | -119 | 56 | -2.12 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 1007 | 444 | 563 | 44.09% | 42.08% | 43.12% | 5.91 pp | -119 | 52 | -2.29 |
| BTC Hourly | nn | NN | 1007 | 443 | 564 | 43.99% | 41.67% | 41.04% | 6.01 pp | -121 | 52 | -2.33 |
| Consolidated Hourly | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 830 | 349 | 481 | 42.05% | 35.00% | 39.79% | 7.95 pp | -132 | 47 | -2.81 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 1007 | 426 | 581 | 42.30% | 36.67% | 40.00% | 7.70 pp | -155 | 52 | -2.98 |
| BTC Daily | rf | RandomForest | 830 | 344 | 486 | 41.45% | 37.08% | 40.62% | 8.55 pp | -142 | 47 | -3.02 |
| Consolidated Hourly | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1007 | 414 | 593 | 41.11% | 35.42% | 38.33% | 8.89 pp | -179 | 52 | -3.44 |
| BTC Daily | xgb | XGBoost | 840 | 331 | 509 | 39.40% | 37.08% | 36.88% | 10.60 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1007 | 477 | 530 | 47.37% | 47.92% | 45.62% | 2.63 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1007 | 469 | 538 | 46.57% | 45.83% | 44.58% | 3.43 pp | -69 | 52 | -1.33 |
| BTC Hourly | rf | RandomForest | 1007 | 444 | 563 | 44.09% | 42.08% | 43.12% | 5.91 pp | -119 | 52 | -2.29 |
| BTC Hourly | nn | NN | 1007 | 443 | 564 | 43.99% | 41.67% | 41.04% | 6.01 pp | -121 | 52 | -2.33 |
| BTC Hourly | lstm | LSTM | 1007 | 426 | 581 | 42.30% | 36.67% | 40.00% | 7.70 pp | -155 | 52 | -2.98 |
| BTC Hourly | xgb | XGBoost | 1007 | 414 | 593 | 41.11% | 35.42% | 38.33% | 8.89 pp | -179 | 52 | -3.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 830 | 396 | 434 | 47.71% | 45.00% | 46.04% | 2.29 pp | -38 | 47 | -0.81 |
| BTC Daily | nn | NN | 830 | 386 | 444 | 46.51% | 45.00% | 45.21% | 3.49 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 830 | 383 | 447 | 46.14% | 37.92% | 45.00% | 3.86 pp | -64 | 47 | -1.36 |
| BTC Daily | lstm | LSTM | 830 | 349 | 481 | 42.05% | 35.00% | 39.79% | 7.95 pp | -132 | 47 | -2.81 |
| BTC Daily | rf | RandomForest | 830 | 344 | 486 | 41.45% | 37.08% | 40.62% | 8.55 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 840 | 331 | 509 | 39.40% | 37.08% | 36.88% | 10.60 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 602 | 289 | 313 | 48.01% | 46.25% | 47.08% | 1.99 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 602 | 287 | 315 | 47.67% | 51.25% | 48.96% | 2.33 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 602 | 282 | 320 | 46.84% | 45.83% | 46.04% | 3.16 pp | -38 | 56 | -0.68 |
| BTC Market Hours | lstm | LSTM | 602 | 258 | 344 | 42.86% | 42.50% | 42.92% | 7.14 pp | -86 | 56 | -1.54 |
| BTC Market Hours | rf | RandomForest | 602 | 257 | 345 | 42.69% | 42.92% | 42.29% | 7.31 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 602 | 256 | 346 | 42.52% | 45.00% | 43.33% | 7.48 pp | -90 | 56 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 655 | 307 | 348 | 46.87% | 48.33% | 47.71% | 3.13 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 655 | 306 | 349 | 46.72% | 48.33% | 47.08% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 655 | 306 | 349 | 46.72% | 48.33% | 48.12% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | rf | RandomForest | 655 | 271 | 384 | 41.37% | 42.08% | 41.25% | 8.63 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 655 | 268 | 387 | 40.92% | 42.50% | 40.42% | 9.08 pp | -119 | 56 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 655 | 268 | 387 | 40.92% | 42.50% | 40.62% | 9.08 pp | -119 | 56 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
