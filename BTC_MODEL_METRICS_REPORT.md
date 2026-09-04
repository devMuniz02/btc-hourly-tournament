# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T02:02:31.198296+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1220 | 932 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1096 | 731 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 764 | 493 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 766 | 547 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 142 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 493 | 237 | 256 | 48.07% | 44.17% | 47.92% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 493 | 233 | 260 | 47.26% | 50.42% | 47.71% | 2.74 pp | -27 | 47 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 721 | 348 | 373 | 48.27% | 46.67% | 47.92% | 1.73 pp | -25 | 43 | -0.58 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 493 | 230 | 263 | 46.65% | 42.92% | 47.29% | 3.35 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 721 | 345 | 376 | 47.85% | 46.67% | 50.21% | 2.15 pp | -31 | 43 | -0.72 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 898 | 430 | 468 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 547 | 254 | 293 | 46.44% | 49.17% | 47.29% | 3.56 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 547 | 252 | 295 | 46.07% | 44.58% | 47.08% | 3.93 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 547 | 251 | 296 | 45.89% | 48.33% | 46.88% | 4.11 pp | -45 | 47 | -0.96 |
| Consolidated Hourly | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 898 | 424 | 474 | 47.22% | 47.50% | 46.46% | 2.78 pp | -50 | 47 | -1.06 |
| BTC Daily | nn | NN | 721 | 335 | 386 | 46.46% | 44.58% | 47.92% | 3.54 pp | -51 | 43 | -1.19 |
| Consolidated Hourly | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 493 | 213 | 280 | 43.20% | 41.25% | 43.33% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 493 | 213 | 280 | 43.20% | 43.75% | 43.54% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 493 | 202 | 291 | 40.97% | 40.42% | 41.04% | 9.03 pp | -89 | 47 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 547 | 227 | 320 | 41.50% | 41.67% | 41.25% | 8.50 pp | -93 | 47 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 898 | 400 | 498 | 44.54% | 44.58% | 42.50% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 898 | 400 | 498 | 44.54% | 45.00% | 44.38% | 5.46 pp | -98 | 47 | -2.09 |
| Consolidated Hourly | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | lstm | LSTM | 721 | 312 | 409 | 43.27% | 37.92% | 42.08% | 6.73 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 547 | 220 | 327 | 40.22% | 37.92% | 40.62% | 9.78 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 547 | 219 | 328 | 40.04% | 40.83% | 39.38% | 9.96 pp | -109 | 47 | -2.32 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 721 | 308 | 413 | 42.72% | 40.83% | 43.54% | 7.28 pp | -105 | 43 | -2.44 |
| BTC Hourly | lstm | LSTM | 898 | 384 | 514 | 42.76% | 39.58% | 42.50% | 7.24 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 898 | 378 | 520 | 42.09% | 42.50% | 42.08% | 7.91 pp | -142 | 47 | -3.02 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 731 | 290 | 441 | 39.67% | 36.25% | 38.54% | 10.33 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours Daily | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 9 | 24 | 27.27% | 27.27% | 27.27% | 22.73 pp | -15 | 3 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 898 | 430 | 468 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| BTC Hourly | transformer | Transformer | 898 | 424 | 474 | 47.22% | 47.50% | 46.46% | 2.78 pp | -50 | 47 | -1.06 |
| BTC Hourly | nn | NN | 898 | 400 | 498 | 44.54% | 44.58% | 42.50% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 898 | 400 | 498 | 44.54% | 45.00% | 44.38% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 898 | 384 | 514 | 42.76% | 39.58% | 42.50% | 7.24 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 898 | 378 | 520 | 42.09% | 42.50% | 42.08% | 7.91 pp | -142 | 47 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 721 | 348 | 373 | 48.27% | 46.67% | 47.92% | 1.73 pp | -25 | 43 | -0.58 |
| BTC Daily | transformer | Transformer | 721 | 345 | 376 | 47.85% | 46.67% | 50.21% | 2.15 pp | -31 | 43 | -0.72 |
| BTC Daily | nn | NN | 721 | 335 | 386 | 46.46% | 44.58% | 47.92% | 3.54 pp | -51 | 43 | -1.19 |
| BTC Daily | lstm | LSTM | 721 | 312 | 409 | 43.27% | 37.92% | 42.08% | 6.73 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 721 | 308 | 413 | 42.72% | 40.83% | 43.54% | 7.28 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 731 | 290 | 441 | 39.67% | 36.25% | 38.54% | 10.33 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 493 | 237 | 256 | 48.07% | 44.17% | 47.92% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 493 | 233 | 260 | 47.26% | 50.42% | 47.71% | 2.74 pp | -27 | 47 | -0.57 |
| BTC Market Hours | transformer | Transformer | 493 | 230 | 263 | 46.65% | 42.92% | 47.29% | 3.35 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 493 | 213 | 280 | 43.20% | 41.25% | 43.33% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 493 | 213 | 280 | 43.20% | 43.75% | 43.54% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 493 | 202 | 291 | 40.97% | 40.42% | 41.04% | 9.03 pp | -89 | 47 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 547 | 254 | 293 | 46.44% | 49.17% | 47.29% | 3.56 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 547 | 252 | 295 | 46.07% | 44.58% | 47.08% | 3.93 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 547 | 251 | 296 | 45.89% | 48.33% | 46.88% | 4.11 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 547 | 227 | 320 | 41.50% | 41.67% | 41.25% | 8.50 pp | -93 | 47 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 547 | 220 | 327 | 40.22% | 37.92% | 40.62% | 9.78 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 547 | 219 | 328 | 40.04% | 40.83% | 39.38% | 9.96 pp | -109 | 47 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 9 | 24 | 27.27% | 27.27% | 27.27% | 22.73 pp | -15 | 3 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
