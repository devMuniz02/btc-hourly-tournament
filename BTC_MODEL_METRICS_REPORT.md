# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T09:22:19.339494+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1225 | 937 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1101 | 736 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 769 | 498 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 771 | 552 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 147 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 146 | 74 | 72 | 50.68% | 50.68% | 50.68% | 0.68 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 146 | 74 | 72 | 50.68% | 50.68% | 50.68% | 0.68 pp | 2 | 11 | 0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 498 | 242 | 256 | 48.59% | 45.42% | 48.33% | 1.41 pp | -14 | 48 | -0.29 |
| Consolidated Hourly | xgb | XGBoost | 146 | 71 | 75 | 48.63% | 48.63% | 48.63% | 1.37 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 146 | 71 | 75 | 48.63% | 48.63% | 48.63% | 1.37 pp | -4 | 11 | -0.36 |
| BTC Market Hours | nn | NN | 498 | 236 | 262 | 47.39% | 50.83% | 47.92% | 2.61 pp | -26 | 48 | -0.54 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 146 | 70 | 76 | 47.95% | 47.95% | 47.95% | 2.05 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 146 | 70 | 76 | 47.95% | 47.95% | 47.95% | 2.05 pp | -6 | 11 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 726 | 350 | 376 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 498 | 234 | 264 | 46.99% | 44.58% | 47.71% | 3.01 pp | -30 | 48 | -0.62 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 726 | 346 | 380 | 47.66% | 46.67% | 49.58% | 2.34 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 552 | 257 | 295 | 46.56% | 49.17% | 47.50% | 3.44 pp | -38 | 48 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 552 | 255 | 297 | 46.20% | 49.58% | 46.88% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 552 | 255 | 297 | 46.20% | 45.00% | 47.50% | 3.80 pp | -42 | 48 | -0.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Daily | nn | NN | 726 | 336 | 390 | 46.28% | 44.58% | 47.29% | 3.72 pp | -54 | 43 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 498 | 215 | 283 | 43.17% | 41.25% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 498 | 215 | 283 | 43.17% | 44.17% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| Consolidated Hourly | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 11 | -1.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 498 | 205 | 293 | 41.16% | 41.67% | 41.25% | 8.84 pp | -88 | 48 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 552 | 229 | 323 | 41.49% | 42.50% | 40.83% | 8.51 pp | -94 | 48 | -1.96 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 552 | 222 | 330 | 40.22% | 38.33% | 40.42% | 9.78 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 552 | 220 | 332 | 39.86% | 41.25% | 38.96% | 10.14 pp | -112 | 48 | -2.33 |
| Consolidated Hourly | transformer | Transformer | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 11 | -2.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 11 | -2.36 |
| BTC Daily | lstm | LSTM | 726 | 312 | 414 | 42.98% | 36.67% | 41.25% | 7.02 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 726 | 310 | 416 | 42.70% | 40.83% | 43.33% | 7.30 pp | -106 | 43 | -2.47 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 736 | 292 | 444 | 39.67% | 37.08% | 38.54% | 10.33 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 726 | 350 | 376 | 48.21% | 45.83% | 47.92% | 1.79 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 726 | 346 | 380 | 47.66% | 46.67% | 49.58% | 2.34 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 726 | 336 | 390 | 46.28% | 44.58% | 47.29% | 3.72 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 726 | 312 | 414 | 42.98% | 36.67% | 41.25% | 7.02 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 726 | 310 | 416 | 42.70% | 40.83% | 43.33% | 7.30 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 736 | 292 | 444 | 39.67% | 37.08% | 38.54% | 10.33 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 498 | 242 | 256 | 48.59% | 45.42% | 48.33% | 1.41 pp | -14 | 48 | -0.29 |
| BTC Market Hours | nn | NN | 498 | 236 | 262 | 47.39% | 50.83% | 47.92% | 2.61 pp | -26 | 48 | -0.54 |
| BTC Market Hours | transformer | Transformer | 498 | 234 | 264 | 46.99% | 44.58% | 47.71% | 3.01 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 498 | 215 | 283 | 43.17% | 41.25% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 498 | 215 | 283 | 43.17% | 44.17% | 43.33% | 6.83 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 498 | 205 | 293 | 41.16% | 41.67% | 41.25% | 8.84 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 552 | 257 | 295 | 46.56% | 49.17% | 47.50% | 3.44 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 552 | 255 | 297 | 46.20% | 49.58% | 46.88% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 552 | 255 | 297 | 46.20% | 45.00% | 47.50% | 3.80 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 552 | 229 | 323 | 41.49% | 42.50% | 40.83% | 8.51 pp | -94 | 48 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 552 | 222 | 330 | 40.22% | 38.33% | 40.42% | 9.78 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 552 | 220 | 332 | 39.86% | 41.25% | 38.96% | 10.14 pp | -112 | 48 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 146 | 74 | 72 | 50.68% | 50.68% | 50.68% | 0.68 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | xgb | XGBoost | 146 | 71 | 75 | 48.63% | 48.63% | 48.63% | 1.37 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 146 | 70 | 76 | 47.95% | 47.95% | 47.95% | 2.05 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | lstm | LSTM | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 11 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 11 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 146 | 74 | 72 | 50.68% | 50.68% | 50.68% | 0.68 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 146 | 71 | 75 | 48.63% | 48.63% | 48.63% | 1.37 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 146 | 70 | 76 | 47.95% | 47.95% | 47.95% | 2.05 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 146 | 66 | 80 | 45.21% | 45.21% | 45.21% | 4.79 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 146 | 65 | 81 | 44.52% | 44.52% | 44.52% | 5.48 pp | -16 | 11 | -1.45 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 146 | 60 | 86 | 41.10% | 41.10% | 41.10% | 8.90 pp | -26 | 11 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
