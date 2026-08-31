# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T22:39:27.443511+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1169 | 881 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1045 | 680 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 21:00:00+00:00 | 671 | 442 | 228 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 21:00:00+00:00 | 673 | 496 | 175 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 96 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 442 | 215 | 227 | 48.64% | 44.58% | 48.64% | 1.36 pp | -12 | 43 | -0.28 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 670 | 327 | 343 | 48.81% | 47.08% | 49.58% | 1.19 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 670 | 324 | 346 | 48.36% | 45.83% | 49.58% | 1.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours | nn | NN | 442 | 209 | 233 | 47.29% | 48.75% | 47.29% | 2.71 pp | -24 | 43 | -0.56 |
| BTC Market Hours | transformer | Transformer | 442 | 202 | 240 | 45.70% | 40.42% | 45.70% | 4.30 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 496 | 228 | 268 | 45.97% | 46.25% | 46.46% | 4.03 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | nn | NN | 496 | 227 | 269 | 45.77% | 43.75% | 46.46% | 4.23 pp | -42 | 43 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 670 | 314 | 356 | 46.87% | 43.75% | 49.38% | 3.13 pp | -42 | 41 | -1.02 |
| BTC Hourly | transformer | Transformer | 847 | 400 | 447 | 47.23% | 47.50% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | transformer | Transformer | 496 | 224 | 272 | 45.16% | 45.00% | 45.00% | 4.84 pp | -48 | 43 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 847 | 398 | 449 | 46.99% | 44.58% | 46.88% | 3.01 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 442 | 190 | 252 | 42.99% | 43.33% | 42.99% | 7.01 pp | -62 | 43 | -1.44 |
| BTC Market Hours | lstm | LSTM | 442 | 189 | 253 | 42.76% | 41.25% | 42.76% | 7.24 pp | -64 | 43 | -1.49 |
| BTC Hourly | nn | NN | 847 | 382 | 465 | 45.10% | 43.75% | 44.58% | 4.90 pp | -83 | 45 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 442 | 178 | 264 | 40.27% | 38.75% | 40.27% | 9.73 pp | -86 | 43 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 43 | -2.05 |
| BTC Daily | lstm | LSTM | 670 | 293 | 377 | 43.73% | 38.75% | 42.92% | 6.27 pp | -84 | 41 | -2.05 |
| BTC Hourly | rf | RandomForest | 847 | 377 | 470 | 44.51% | 43.33% | 43.96% | 5.49 pp | -93 | 45 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 496 | 198 | 298 | 39.92% | 37.92% | 40.62% | 10.08 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 670 | 286 | 384 | 42.69% | 40.42% | 43.75% | 7.31 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 496 | 194 | 302 | 39.11% | 36.25% | 39.17% | 10.89 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 847 | 363 | 484 | 42.86% | 39.58% | 42.29% | 7.14 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 847 | 356 | 491 | 42.03% | 40.00% | 42.29% | 7.97 pp | -135 | 45 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 680 | 270 | 410 | 39.71% | 34.58% | 39.58% | 10.29 pp | -140 | 41 | -3.41 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 847 | 400 | 447 | 47.23% | 47.50% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 847 | 398 | 449 | 46.99% | 44.58% | 46.88% | 3.01 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 847 | 382 | 465 | 45.10% | 43.75% | 44.58% | 4.90 pp | -83 | 45 | -1.84 |
| BTC Hourly | rf | RandomForest | 847 | 377 | 470 | 44.51% | 43.33% | 43.96% | 5.49 pp | -93 | 45 | -2.07 |
| BTC Hourly | lstm | LSTM | 847 | 363 | 484 | 42.86% | 39.58% | 42.29% | 7.14 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 847 | 356 | 491 | 42.03% | 40.00% | 42.29% | 7.97 pp | -135 | 45 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 670 | 327 | 343 | 48.81% | 47.08% | 49.58% | 1.19 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 670 | 324 | 346 | 48.36% | 45.83% | 49.58% | 1.64 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 670 | 314 | 356 | 46.87% | 43.75% | 49.38% | 3.13 pp | -42 | 41 | -1.02 |
| BTC Daily | lstm | LSTM | 670 | 293 | 377 | 43.73% | 38.75% | 42.92% | 6.27 pp | -84 | 41 | -2.05 |
| BTC Daily | rf | RandomForest | 670 | 286 | 384 | 42.69% | 40.42% | 43.75% | 7.31 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 680 | 270 | 410 | 39.71% | 34.58% | 39.58% | 10.29 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 442 | 215 | 227 | 48.64% | 44.58% | 48.64% | 1.36 pp | -12 | 43 | -0.28 |
| BTC Market Hours | nn | NN | 442 | 209 | 233 | 47.29% | 48.75% | 47.29% | 2.71 pp | -24 | 43 | -0.56 |
| BTC Market Hours | transformer | Transformer | 442 | 202 | 240 | 45.70% | 40.42% | 45.70% | 4.30 pp | -38 | 43 | -0.88 |
| BTC Market Hours | rf | RandomForest | 442 | 190 | 252 | 42.99% | 43.33% | 42.99% | 7.01 pp | -62 | 43 | -1.44 |
| BTC Market Hours | lstm | LSTM | 442 | 189 | 253 | 42.76% | 41.25% | 42.76% | 7.24 pp | -64 | 43 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 442 | 178 | 264 | 40.27% | 38.75% | 40.27% | 9.73 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 496 | 228 | 268 | 45.97% | 46.25% | 46.46% | 4.03 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | nn | NN | 496 | 227 | 269 | 45.77% | 43.75% | 46.46% | 4.23 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 496 | 224 | 272 | 45.16% | 45.00% | 45.00% | 4.84 pp | -48 | 43 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 496 | 198 | 298 | 39.92% | 37.92% | 40.62% | 10.08 pp | -100 | 43 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 496 | 194 | 302 | 39.11% | 36.25% | 39.17% | 10.89 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
