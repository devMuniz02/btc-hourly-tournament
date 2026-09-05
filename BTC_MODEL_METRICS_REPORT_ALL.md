# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T02:42:14.867239+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1237 | 949 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1112 | 747 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 793 | 509 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 795 | 563 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 509 | 246 | 263 | 48.33% | 45.00% | 48.33% | 1.67 pp | -17 | 49 | -0.35 |
| BTC Market Hours | transformer | Transformer | 509 | 244 | 265 | 47.94% | 46.25% | 48.33% | 2.06 pp | -21 | 49 | -0.43 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 737 | 356 | 381 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 44 | -0.57 |
| BTC Market Hours Daily | transformer | Transformer | 563 | 267 | 296 | 47.42% | 50.83% | 48.75% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours | nn | NN | 509 | 239 | 270 | 46.95% | 49.17% | 47.92% | 3.05 pp | -31 | 49 | -0.63 |
| BTC Daily | transformer | Transformer | 737 | 351 | 386 | 47.63% | 46.25% | 49.58% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 915 | 438 | 477 | 47.87% | 50.00% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | nn | NN | 563 | 261 | 302 | 46.36% | 45.83% | 47.71% | 3.64 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 563 | 260 | 303 | 46.18% | 48.75% | 46.67% | 3.82 pp | -43 | 48 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 915 | 432 | 483 | 47.21% | 47.50% | 46.25% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 737 | 341 | 396 | 46.27% | 44.17% | 46.88% | 3.73 pp | -55 | 44 | -1.25 |
| BTC Market Hours | lstm | LSTM | 509 | 221 | 288 | 43.42% | 42.92% | 43.54% | 6.58 pp | -67 | 49 | -1.37 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours | rf | RandomForest | 509 | 219 | 290 | 43.03% | 43.75% | 43.33% | 6.97 pp | -71 | 49 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 509 | 209 | 300 | 41.06% | 42.50% | 41.67% | 8.94 pp | -91 | 49 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 563 | 234 | 329 | 41.56% | 42.50% | 40.42% | 8.44 pp | -95 | 48 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 915 | 407 | 508 | 44.48% | 43.75% | 44.17% | 5.52 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 915 | 406 | 509 | 44.37% | 42.92% | 42.08% | 5.63 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 563 | 229 | 334 | 40.67% | 39.58% | 41.04% | 9.33 pp | -105 | 48 | -2.19 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Daily | lstm | LSTM | 737 | 317 | 420 | 43.01% | 37.08% | 41.25% | 6.99 pp | -103 | 44 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 563 | 225 | 338 | 39.96% | 40.42% | 38.96% | 10.04 pp | -113 | 48 | -2.35 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 737 | 312 | 425 | 42.33% | 40.00% | 42.71% | 7.67 pp | -113 | 44 | -2.57 |
| BTC Hourly | lstm | LSTM | 915 | 391 | 524 | 42.73% | 39.17% | 41.46% | 7.27 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 915 | 382 | 533 | 41.75% | 39.58% | 40.00% | 8.25 pp | -151 | 48 | -3.15 |
| BTC Daily | xgb | XGBoost | 747 | 295 | 452 | 39.49% | 36.25% | 37.71% | 10.51 pp | -157 | 44 | -3.57 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 915 | 438 | 477 | 47.87% | 50.00% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 915 | 432 | 483 | 47.21% | 47.50% | 46.25% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Hourly | rf | RandomForest | 915 | 407 | 508 | 44.48% | 43.75% | 44.17% | 5.52 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 915 | 406 | 509 | 44.37% | 42.92% | 42.08% | 5.63 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 915 | 391 | 524 | 42.73% | 39.17% | 41.46% | 7.27 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 915 | 382 | 533 | 41.75% | 39.58% | 40.00% | 8.25 pp | -151 | 48 | -3.15 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 737 | 356 | 381 | 48.30% | 47.08% | 48.12% | 1.70 pp | -25 | 44 | -0.57 |
| BTC Daily | transformer | Transformer | 737 | 351 | 386 | 47.63% | 46.25% | 49.58% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 737 | 341 | 396 | 46.27% | 44.17% | 46.88% | 3.73 pp | -55 | 44 | -1.25 |
| BTC Daily | lstm | LSTM | 737 | 317 | 420 | 43.01% | 37.08% | 41.25% | 6.99 pp | -103 | 44 | -2.34 |
| BTC Daily | rf | RandomForest | 737 | 312 | 425 | 42.33% | 40.00% | 42.71% | 7.67 pp | -113 | 44 | -2.57 |
| BTC Daily | xgb | XGBoost | 747 | 295 | 452 | 39.49% | 36.25% | 37.71% | 10.51 pp | -157 | 44 | -3.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 509 | 246 | 263 | 48.33% | 45.00% | 48.33% | 1.67 pp | -17 | 49 | -0.35 |
| BTC Market Hours | transformer | Transformer | 509 | 244 | 265 | 47.94% | 46.25% | 48.33% | 2.06 pp | -21 | 49 | -0.43 |
| BTC Market Hours | nn | NN | 509 | 239 | 270 | 46.95% | 49.17% | 47.92% | 3.05 pp | -31 | 49 | -0.63 |
| BTC Market Hours | lstm | LSTM | 509 | 221 | 288 | 43.42% | 42.92% | 43.54% | 6.58 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 509 | 219 | 290 | 43.03% | 43.75% | 43.33% | 6.97 pp | -71 | 49 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 509 | 209 | 300 | 41.06% | 42.50% | 41.67% | 8.94 pp | -91 | 49 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 563 | 267 | 296 | 47.42% | 50.83% | 48.75% | 2.58 pp | -29 | 48 | -0.60 |
| BTC Market Hours Daily | nn | NN | 563 | 261 | 302 | 46.36% | 45.83% | 47.71% | 3.64 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 563 | 260 | 303 | 46.18% | 48.75% | 46.67% | 3.82 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 563 | 234 | 329 | 41.56% | 42.50% | 40.42% | 8.44 pp | -95 | 48 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 563 | 229 | 334 | 40.67% | 39.58% | 41.04% | 9.33 pp | -105 | 48 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 563 | 225 | 338 | 39.96% | 40.42% | 38.96% | 10.04 pp | -113 | 48 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
