# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T21:03:04.522219+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 960 | 602 | 357 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 962 | 656 | 304 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 602 | 289 | 313 | 48.01% | 46.25% | 47.08% | 1.99 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 602 | 287 | 315 | 47.67% | 51.25% | 48.96% | 2.33 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 602 | 282 | 320 | 46.84% | 45.83% | 46.04% | 3.16 pp | -38 | 56 | -0.68 |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | nn | NN | 656 | 307 | 349 | 46.80% | 48.33% | 47.50% | 3.20 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 656 | 306 | 350 | 46.65% | 48.33% | 46.88% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 656 | 306 | 350 | 46.65% | 47.92% | 47.92% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Daily | mlp_sklearn | MLPClassifier | 830 | 396 | 434 | 47.71% | 45.00% | 46.04% | 2.29 pp | -38 | 47 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1007 | 478 | 529 | 47.47% | 48.33% | 45.83% | 2.53 pp | -51 | 52 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 830 | 386 | 444 | 46.51% | 45.00% | 45.21% | 3.49 pp | -58 | 47 | -1.23 |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1007 | 470 | 537 | 46.67% | 46.25% | 44.79% | 3.33 pp | -67 | 52 | -1.29 |
| BTC Daily | transformer | Transformer | 830 | 383 | 447 | 46.14% | 37.92% | 45.00% | 3.86 pp | -64 | 47 | -1.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| BTC Market Hours | lstm | LSTM | 602 | 258 | 344 | 42.86% | 42.50% | 42.92% | 7.14 pp | -86 | 56 | -1.54 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 602 | 257 | 345 | 42.69% | 42.92% | 42.29% | 7.31 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 602 | 256 | 346 | 42.52% | 45.00% | 43.33% | 7.48 pp | -90 | 56 | -1.61 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 656 | 271 | 385 | 41.31% | 42.08% | 41.25% | 8.69 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 656 | 268 | 388 | 40.85% | 42.50% | 40.42% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 656 | 268 | 388 | 40.85% | 42.50% | 40.62% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1007 | 444 | 563 | 44.09% | 42.08% | 41.25% | 5.91 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1007 | 444 | 563 | 44.09% | 41.67% | 43.12% | 5.91 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| BTC Daily | lstm | LSTM | 830 | 349 | 481 | 42.05% | 35.00% | 39.79% | 7.95 pp | -132 | 47 | -2.81 |
| BTC Hourly | lstm | LSTM | 1007 | 426 | 581 | 42.30% | 36.25% | 40.00% | 7.70 pp | -155 | 52 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| BTC Daily | rf | RandomForest | 830 | 344 | 486 | 41.45% | 37.08% | 40.62% | 8.55 pp | -142 | 47 | -3.02 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1007 | 414 | 593 | 41.11% | 35.42% | 38.33% | 8.89 pp | -179 | 52 | -3.44 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 840 | 331 | 509 | 39.40% | 37.08% | 36.88% | 10.60 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1007 | 478 | 529 | 47.47% | 48.33% | 45.83% | 2.53 pp | -51 | 52 | -0.98 |
| BTC Hourly | transformer | Transformer | 1007 | 470 | 537 | 46.67% | 46.25% | 44.79% | 3.33 pp | -67 | 52 | -1.29 |
| BTC Hourly | nn | NN | 1007 | 444 | 563 | 44.09% | 42.08% | 41.25% | 5.91 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1007 | 444 | 563 | 44.09% | 41.67% | 43.12% | 5.91 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 1007 | 426 | 581 | 42.30% | 36.25% | 40.00% | 7.70 pp | -155 | 52 | -2.98 |
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
| BTC Market Hours Daily | nn | NN | 656 | 307 | 349 | 46.80% | 48.33% | 47.50% | 3.20 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 656 | 306 | 350 | 46.65% | 48.33% | 46.88% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 656 | 306 | 350 | 46.65% | 47.92% | 47.92% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 656 | 271 | 385 | 41.31% | 42.08% | 41.25% | 8.69 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 656 | 268 | 388 | 40.85% | 42.50% | 40.42% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 656 | 268 | 388 | 40.85% | 42.50% | 40.62% | 9.15 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
