# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T13:21:52.231105+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1180 | 892 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1055 | 690 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 12:00:00+00:00 | 685 | 452 | 232 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 12:00:00+00:00 | 687 | 506 | 179 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 452 | 219 | 233 | 48.45% | 44.58% | 48.45% | 1.55 pp | -14 | 44 | -0.32 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 680 | 333 | 347 | 48.97% | 47.50% | 49.58% | 1.03 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 680 | 328 | 352 | 48.24% | 46.25% | 49.38% | 1.76 pp | -24 | 41 | -0.59 |
| BTC Market Hours | nn | NN | 452 | 213 | 239 | 47.12% | 48.33% | 47.12% | 2.88 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 452 | 209 | 243 | 46.24% | 40.00% | 46.24% | 3.76 pp | -34 | 44 | -0.77 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 506 | 232 | 274 | 45.85% | 46.25% | 46.25% | 4.15 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 506 | 231 | 275 | 45.65% | 43.33% | 46.46% | 4.35 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 506 | 230 | 276 | 45.45% | 46.25% | 45.62% | 4.55 pp | -46 | 44 | -1.05 |
| BTC Daily | nn | NN | 680 | 318 | 362 | 46.76% | 43.33% | 48.96% | 3.24 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 858 | 404 | 454 | 47.09% | 45.00% | 46.67% | 2.91 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 858 | 403 | 455 | 46.97% | 46.67% | 46.46% | 3.03 pp | -52 | 46 | -1.13 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| BTC Market Hours | rf | RandomForest | 452 | 196 | 256 | 43.36% | 43.75% | 43.36% | 6.64 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 452 | 191 | 261 | 42.26% | 39.58% | 42.26% | 7.74 pp | -70 | 44 | -1.59 |
| BTC Hourly | nn | NN | 858 | 387 | 471 | 45.10% | 45.42% | 44.38% | 4.90 pp | -84 | 46 | -1.83 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 506 | 210 | 296 | 41.50% | 41.25% | 41.67% | 8.50 pp | -86 | 44 | -1.95 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 452 | 182 | 270 | 40.27% | 38.33% | 40.27% | 9.73 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 858 | 382 | 476 | 44.52% | 43.33% | 43.75% | 5.48 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 680 | 297 | 383 | 43.68% | 39.17% | 42.71% | 6.32 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 506 | 202 | 304 | 39.92% | 37.50% | 40.62% | 10.08 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 680 | 292 | 388 | 42.94% | 40.83% | 43.54% | 7.06 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 506 | 198 | 308 | 39.13% | 35.83% | 38.75% | 10.87 pp | -110 | 44 | -2.50 |
| BTC Hourly | lstm | LSTM | 858 | 365 | 493 | 42.54% | 37.50% | 41.67% | 7.46 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 858 | 361 | 497 | 42.07% | 40.00% | 42.29% | 7.93 pp | -136 | 46 | -2.96 |
| BTC Daily | xgb | XGBoost | 690 | 274 | 416 | 39.71% | 35.00% | 39.58% | 10.29 pp | -142 | 41 | -3.46 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 858 | 404 | 454 | 47.09% | 45.00% | 46.67% | 2.91 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 858 | 403 | 455 | 46.97% | 46.67% | 46.46% | 3.03 pp | -52 | 46 | -1.13 |
| BTC Hourly | nn | NN | 858 | 387 | 471 | 45.10% | 45.42% | 44.38% | 4.90 pp | -84 | 46 | -1.83 |
| BTC Hourly | rf | RandomForest | 858 | 382 | 476 | 44.52% | 43.33% | 43.75% | 5.48 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 858 | 365 | 493 | 42.54% | 37.50% | 41.67% | 7.46 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 858 | 361 | 497 | 42.07% | 40.00% | 42.29% | 7.93 pp | -136 | 46 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 680 | 333 | 347 | 48.97% | 47.50% | 49.58% | 1.03 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 680 | 328 | 352 | 48.24% | 46.25% | 49.38% | 1.76 pp | -24 | 41 | -0.59 |
| BTC Daily | nn | NN | 680 | 318 | 362 | 46.76% | 43.33% | 48.96% | 3.24 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 680 | 297 | 383 | 43.68% | 39.17% | 42.71% | 6.32 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 680 | 292 | 388 | 42.94% | 40.83% | 43.54% | 7.06 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 690 | 274 | 416 | 39.71% | 35.00% | 39.58% | 10.29 pp | -142 | 41 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 452 | 219 | 233 | 48.45% | 44.58% | 48.45% | 1.55 pp | -14 | 44 | -0.32 |
| BTC Market Hours | nn | NN | 452 | 213 | 239 | 47.12% | 48.33% | 47.12% | 2.88 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 452 | 209 | 243 | 46.24% | 40.00% | 46.24% | 3.76 pp | -34 | 44 | -0.77 |
| BTC Market Hours | rf | RandomForest | 452 | 196 | 256 | 43.36% | 43.75% | 43.36% | 6.64 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 452 | 191 | 261 | 42.26% | 39.58% | 42.26% | 7.74 pp | -70 | 44 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 452 | 182 | 270 | 40.27% | 38.33% | 40.27% | 9.73 pp | -88 | 44 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 506 | 232 | 274 | 45.85% | 46.25% | 46.25% | 4.15 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 506 | 231 | 275 | 45.65% | 43.33% | 46.46% | 4.35 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 506 | 230 | 276 | 45.45% | 46.25% | 45.62% | 4.55 pp | -46 | 44 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 506 | 210 | 296 | 41.50% | 41.25% | 41.67% | 8.50 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 506 | 202 | 304 | 39.92% | 37.50% | 40.62% | 10.08 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 506 | 198 | 308 | 39.13% | 35.83% | 38.75% | 10.87 pp | -110 | 44 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
