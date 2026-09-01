# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T09:17:24.208677+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1177 | 889 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1053 | 688 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 682 | 450 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 684 | 504 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 450 | 218 | 232 | 48.44% | 44.58% | 48.44% | 1.56 pp | -14 | 44 | -0.32 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 678 | 332 | 346 | 48.97% | 47.92% | 49.79% | 1.03 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 678 | 328 | 350 | 48.38% | 46.67% | 49.38% | 1.62 pp | -22 | 41 | -0.54 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 450 | 211 | 239 | 46.89% | 47.50% | 46.89% | 3.11 pp | -28 | 44 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 450 | 207 | 243 | 46.00% | 40.00% | 46.00% | 4.00 pp | -36 | 44 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 504 | 231 | 273 | 45.83% | 46.25% | 46.04% | 4.17 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 504 | 230 | 274 | 45.63% | 43.33% | 46.46% | 4.37 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 678 | 318 | 360 | 46.90% | 43.33% | 49.17% | 3.10 pp | -42 | 41 | -1.02 |
| BTC Hourly | transformer | Transformer | 855 | 403 | 452 | 47.13% | 47.08% | 46.88% | 2.87 pp | -49 | 46 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 504 | 228 | 276 | 45.24% | 45.83% | 45.21% | 4.76 pp | -48 | 44 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 855 | 402 | 453 | 47.02% | 44.58% | 46.67% | 2.98 pp | -51 | 46 | -1.11 |
| BTC Market Hours | rf | RandomForest | 450 | 194 | 256 | 43.11% | 42.92% | 43.11% | 6.89 pp | -62 | 44 | -1.41 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 450 | 191 | 259 | 42.44% | 39.58% | 42.44% | 7.56 pp | -68 | 44 | -1.55 |
| BTC Hourly | nn | NN | 855 | 386 | 469 | 45.15% | 45.42% | 44.38% | 4.85 pp | -83 | 46 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 504 | 209 | 295 | 41.47% | 41.67% | 41.67% | 8.53 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 450 | 180 | 270 | 40.00% | 37.50% | 40.00% | 10.00 pp | -90 | 44 | -2.05 |
| BTC Hourly | rf | RandomForest | 855 | 380 | 475 | 44.44% | 42.92% | 43.75% | 5.56 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 678 | 295 | 383 | 43.51% | 38.33% | 42.50% | 6.49 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 504 | 201 | 303 | 39.88% | 37.50% | 40.42% | 10.12 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 678 | 291 | 387 | 42.92% | 40.83% | 43.54% | 7.08 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 504 | 197 | 307 | 39.09% | 36.25% | 38.54% | 10.91 pp | -110 | 44 | -2.50 |
| BTC Hourly | lstm | LSTM | 855 | 364 | 491 | 42.57% | 37.92% | 41.88% | 7.43 pp | -127 | 46 | -2.76 |
| BTC Hourly | xgb | XGBoost | 855 | 359 | 496 | 41.99% | 39.58% | 42.08% | 8.01 pp | -137 | 46 | -2.98 |
| BTC Daily | xgb | XGBoost | 688 | 273 | 415 | 39.68% | 35.00% | 39.58% | 10.32 pp | -142 | 41 | -3.46 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 855 | 403 | 452 | 47.13% | 47.08% | 46.88% | 2.87 pp | -49 | 46 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 855 | 402 | 453 | 47.02% | 44.58% | 46.67% | 2.98 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 855 | 386 | 469 | 45.15% | 45.42% | 44.38% | 4.85 pp | -83 | 46 | -1.80 |
| BTC Hourly | rf | RandomForest | 855 | 380 | 475 | 44.44% | 42.92% | 43.75% | 5.56 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 855 | 364 | 491 | 42.57% | 37.92% | 41.88% | 7.43 pp | -127 | 46 | -2.76 |
| BTC Hourly | xgb | XGBoost | 855 | 359 | 496 | 41.99% | 39.58% | 42.08% | 8.01 pp | -137 | 46 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 678 | 332 | 346 | 48.97% | 47.92% | 49.79% | 1.03 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 678 | 328 | 350 | 48.38% | 46.67% | 49.38% | 1.62 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 678 | 318 | 360 | 46.90% | 43.33% | 49.17% | 3.10 pp | -42 | 41 | -1.02 |
| BTC Daily | lstm | LSTM | 678 | 295 | 383 | 43.51% | 38.33% | 42.50% | 6.49 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 678 | 291 | 387 | 42.92% | 40.83% | 43.54% | 7.08 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 688 | 273 | 415 | 39.68% | 35.00% | 39.58% | 10.32 pp | -142 | 41 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 450 | 218 | 232 | 48.44% | 44.58% | 48.44% | 1.56 pp | -14 | 44 | -0.32 |
| BTC Market Hours | nn | NN | 450 | 211 | 239 | 46.89% | 47.50% | 46.89% | 3.11 pp | -28 | 44 | -0.64 |
| BTC Market Hours | transformer | Transformer | 450 | 207 | 243 | 46.00% | 40.00% | 46.00% | 4.00 pp | -36 | 44 | -0.82 |
| BTC Market Hours | rf | RandomForest | 450 | 194 | 256 | 43.11% | 42.92% | 43.11% | 6.89 pp | -62 | 44 | -1.41 |
| BTC Market Hours | lstm | LSTM | 450 | 191 | 259 | 42.44% | 39.58% | 42.44% | 7.56 pp | -68 | 44 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 450 | 180 | 270 | 40.00% | 37.50% | 40.00% | 10.00 pp | -90 | 44 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 504 | 231 | 273 | 45.83% | 46.25% | 46.04% | 4.17 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 504 | 230 | 274 | 45.63% | 43.33% | 46.46% | 4.37 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 504 | 228 | 276 | 45.24% | 45.83% | 45.21% | 4.76 pp | -48 | 44 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 504 | 209 | 295 | 41.47% | 41.67% | 41.67% | 8.53 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 504 | 201 | 303 | 39.88% | 37.50% | 40.42% | 10.12 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 504 | 197 | 307 | 39.09% | 36.25% | 38.54% | 10.91 pp | -110 | 44 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
