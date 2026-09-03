# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T17:00:42.900704+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1214 | 926 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1090 | 725 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 750 | 487 | 262 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 16:00:00+00:00 | 752 | 541 | 209 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 136 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 347 | 368 | 48.53% | 47.08% | 48.75% | 1.47 pp | -21 | 43 | -0.49 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 715 | 341 | 374 | 47.69% | 45.83% | 49.79% | 2.31 pp | -33 | 43 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 541 | 251 | 290 | 46.40% | 49.58% | 47.29% | 3.60 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 892 | 425 | 467 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 541 | 248 | 293 | 45.84% | 47.92% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 541 | 248 | 293 | 45.84% | 44.58% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 892 | 423 | 469 | 47.42% | 48.75% | 47.29% | 2.58 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 715 | 331 | 384 | 46.29% | 43.75% | 47.71% | 3.71 pp | -53 | 43 | -1.23 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |
| BTC Hourly | nn | NN | 892 | 399 | 493 | 44.73% | 45.83% | 42.71% | 5.27 pp | -94 | 47 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 541 | 223 | 318 | 41.22% | 41.67% | 41.46% | 8.78 pp | -95 | 47 | -2.02 |
| BTC Hourly | rf | RandomForest | 892 | 398 | 494 | 44.62% | 45.42% | 44.17% | 5.38 pp | -96 | 47 | -2.04 |
| BTC Daily | lstm | LSTM | 715 | 309 | 406 | 43.22% | 37.92% | 42.50% | 6.78 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 541 | 217 | 324 | 40.11% | 37.92% | 40.62% | 9.89 pp | -107 | 47 | -2.28 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 541 | 215 | 326 | 39.74% | 39.58% | 39.58% | 10.26 pp | -111 | 47 | -2.36 |
| BTC Daily | rf | RandomForest | 715 | 306 | 409 | 42.80% | 41.25% | 43.54% | 7.20 pp | -103 | 43 | -2.40 |
| BTC Hourly | lstm | LSTM | 892 | 382 | 510 | 42.83% | 39.17% | 42.08% | 7.17 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 892 | 377 | 515 | 42.26% | 42.92% | 42.08% | 7.74 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 725 | 287 | 438 | 39.59% | 35.00% | 38.75% | 10.41 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 892 | 425 | 467 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 892 | 423 | 469 | 47.42% | 48.75% | 47.29% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 892 | 399 | 493 | 44.73% | 45.83% | 42.71% | 5.27 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 892 | 398 | 494 | 44.62% | 45.42% | 44.17% | 5.38 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 892 | 382 | 510 | 42.83% | 39.17% | 42.08% | 7.17 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 892 | 377 | 515 | 42.26% | 42.92% | 42.08% | 7.74 pp | -138 | 47 | -2.94 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 347 | 368 | 48.53% | 47.08% | 48.75% | 1.47 pp | -21 | 43 | -0.49 |
| BTC Daily | transformer | Transformer | 715 | 341 | 374 | 47.69% | 45.83% | 49.79% | 2.31 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 715 | 331 | 384 | 46.29% | 43.75% | 47.71% | 3.71 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 715 | 309 | 406 | 43.22% | 37.92% | 42.50% | 6.78 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 715 | 306 | 409 | 42.80% | 41.25% | 43.54% | 7.20 pp | -103 | 43 | -2.40 |
| BTC Daily | xgb | XGBoost | 725 | 287 | 438 | 39.59% | 35.00% | 38.75% | 10.41 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 541 | 251 | 290 | 46.40% | 49.58% | 47.29% | 3.60 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 541 | 248 | 293 | 45.84% | 47.92% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 541 | 248 | 293 | 45.84% | 44.58% | 46.88% | 4.16 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 541 | 223 | 318 | 41.22% | 41.67% | 41.46% | 8.78 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 541 | 217 | 324 | 40.11% | 37.92% | 40.62% | 9.89 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 541 | 215 | 326 | 39.74% | 39.58% | 39.58% | 10.26 pp | -111 | 47 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
