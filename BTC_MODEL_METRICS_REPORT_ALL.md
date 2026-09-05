# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T21:47:01.468440+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1250 | 962 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1126 | 761 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 20:00:00+00:00 | 816 | 523 | 292 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 20:00:00+00:00 | 818 | 577 | 239 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 169 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 169 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 47 | 122 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 18:00:00+00:00 | 169 | 47 | 122 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 523 | 254 | 269 | 48.57% | 45.42% | 48.54% | 1.43 pp | -15 | 50 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| BTC Market Hours | transformer | Transformer | 523 | 251 | 272 | 47.99% | 47.92% | 48.54% | 2.01 pp | -21 | 50 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 751 | 366 | 385 | 48.74% | 47.92% | 48.96% | 1.26 pp | -19 | 44 | -0.43 |
| BTC Market Hours Daily | transformer | Transformer | 577 | 275 | 302 | 47.66% | 50.83% | 48.96% | 2.34 pp | -27 | 50 | -0.54 |
| BTC Market Hours | nn | NN | 523 | 247 | 276 | 47.23% | 50.42% | 48.54% | 2.77 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 577 | 270 | 307 | 46.79% | 46.25% | 48.12% | 3.21 pp | -37 | 50 | -0.74 |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 751 | 356 | 395 | 47.40% | 44.58% | 48.96% | 2.60 pp | -39 | 44 | -0.89 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 928 | 442 | 486 | 47.63% | 48.75% | 46.67% | 2.37 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 577 | 266 | 311 | 46.10% | 50.00% | 46.67% | 3.90 pp | -45 | 50 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 928 | 436 | 492 | 46.98% | 46.25% | 45.42% | 3.02 pp | -56 | 49 | -1.14 |
| BTC Daily | nn | NN | 751 | 349 | 402 | 46.47% | 44.58% | 46.88% | 3.53 pp | -53 | 44 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 523 | 228 | 295 | 43.59% | 42.92% | 44.17% | 6.41 pp | -67 | 50 | -1.34 |
| BTC Market Hours | rf | RandomForest | 523 | 225 | 298 | 43.02% | 45.00% | 43.75% | 6.98 pp | -73 | 50 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 523 | 217 | 306 | 41.49% | 43.33% | 42.08% | 8.51 pp | -89 | 50 | -1.78 |
| Consolidated Hourly | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 577 | 240 | 337 | 41.59% | 43.33% | 41.25% | 8.41 pp | -97 | 50 | -1.94 |
| BTC Hourly | rf | RandomForest | 928 | 414 | 514 | 44.61% | 44.58% | 44.38% | 5.39 pp | -100 | 49 | -2.04 |
| Consolidated Hourly | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 577 | 236 | 341 | 40.90% | 41.25% | 40.62% | 9.10 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 928 | 411 | 517 | 44.29% | 42.08% | 42.08% | 5.71 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 577 | 231 | 346 | 40.03% | 41.67% | 39.38% | 9.97 pp | -115 | 50 | -2.30 |
| BTC Daily | lstm | LSTM | 751 | 319 | 432 | 42.48% | 35.42% | 40.83% | 7.52 pp | -113 | 44 | -2.57 |
| BTC Daily | rf | RandomForest | 751 | 316 | 435 | 42.08% | 38.33% | 42.08% | 7.92 pp | -119 | 44 | -2.70 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 928 | 396 | 532 | 42.67% | 37.50% | 41.46% | 7.33 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 928 | 388 | 540 | 41.81% | 39.58% | 40.62% | 8.19 pp | -152 | 49 | -3.10 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 761 | 298 | 463 | 39.16% | 35.83% | 36.88% | 10.84 pp | -165 | 44 | -3.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 928 | 442 | 486 | 47.63% | 48.75% | 46.67% | 2.37 pp | -44 | 49 | -0.90 |
| BTC Hourly | transformer | Transformer | 928 | 436 | 492 | 46.98% | 46.25% | 45.42% | 3.02 pp | -56 | 49 | -1.14 |
| BTC Hourly | rf | RandomForest | 928 | 414 | 514 | 44.61% | 44.58% | 44.38% | 5.39 pp | -100 | 49 | -2.04 |
| BTC Hourly | nn | NN | 928 | 411 | 517 | 44.29% | 42.08% | 42.08% | 5.71 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 928 | 396 | 532 | 42.67% | 37.50% | 41.46% | 7.33 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 928 | 388 | 540 | 41.81% | 39.58% | 40.62% | 8.19 pp | -152 | 49 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 751 | 366 | 385 | 48.74% | 47.92% | 48.96% | 1.26 pp | -19 | 44 | -0.43 |
| BTC Daily | transformer | Transformer | 751 | 356 | 395 | 47.40% | 44.58% | 48.96% | 2.60 pp | -39 | 44 | -0.89 |
| BTC Daily | nn | NN | 751 | 349 | 402 | 46.47% | 44.58% | 46.88% | 3.53 pp | -53 | 44 | -1.20 |
| BTC Daily | lstm | LSTM | 751 | 319 | 432 | 42.48% | 35.42% | 40.83% | 7.52 pp | -113 | 44 | -2.57 |
| BTC Daily | rf | RandomForest | 751 | 316 | 435 | 42.08% | 38.33% | 42.08% | 7.92 pp | -119 | 44 | -2.70 |
| BTC Daily | xgb | XGBoost | 761 | 298 | 463 | 39.16% | 35.83% | 36.88% | 10.84 pp | -165 | 44 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 523 | 254 | 269 | 48.57% | 45.42% | 48.54% | 1.43 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 523 | 251 | 272 | 47.99% | 47.92% | 48.54% | 2.01 pp | -21 | 50 | -0.42 |
| BTC Market Hours | nn | NN | 523 | 247 | 276 | 47.23% | 50.42% | 48.54% | 2.77 pp | -29 | 50 | -0.58 |
| BTC Market Hours | lstm | LSTM | 523 | 228 | 295 | 43.59% | 42.92% | 44.17% | 6.41 pp | -67 | 50 | -1.34 |
| BTC Market Hours | rf | RandomForest | 523 | 225 | 298 | 43.02% | 45.00% | 43.75% | 6.98 pp | -73 | 50 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 523 | 217 | 306 | 41.49% | 43.33% | 42.08% | 8.51 pp | -89 | 50 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 577 | 275 | 302 | 47.66% | 50.83% | 48.96% | 2.34 pp | -27 | 50 | -0.54 |
| BTC Market Hours Daily | nn | NN | 577 | 270 | 307 | 46.79% | 46.25% | 48.12% | 3.21 pp | -37 | 50 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 577 | 266 | 311 | 46.10% | 50.00% | 46.67% | 3.90 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 577 | 240 | 337 | 41.59% | 43.33% | 41.25% | 8.41 pp | -97 | 50 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 577 | 236 | 341 | 40.90% | 41.25% | 40.62% | 9.10 pp | -105 | 50 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 577 | 231 | 346 | 40.03% | 41.67% | 39.38% | 9.97 pp | -115 | 50 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 85 | 84 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
