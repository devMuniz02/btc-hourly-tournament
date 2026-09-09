# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T13:49:03.538501+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1309 | 1021 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1185 | 820 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 12:00:00+00:00 | 919 | 582 | 336 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 12:00:00+00:00 | 921 | 636 | 283 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 226 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 582 | 283 | 299 | 48.63% | 47.08% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 582 | 281 | 301 | 48.28% | 52.92% | 50.00% | 1.72 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 582 | 275 | 307 | 47.25% | 47.50% | 47.08% | 2.75 pp | -32 | 54 | -0.59 |
| BTC Daily | mlp_sklearn | MLPClassifier | 810 | 390 | 420 | 48.15% | 46.25% | 47.29% | 1.85 pp | -30 | 47 | -0.64 |
| Consolidated Hourly | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | nn | NN | 636 | 300 | 336 | 47.17% | 48.75% | 48.54% | 2.83 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 636 | 299 | 337 | 47.01% | 50.00% | 47.29% | 2.99 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 636 | 298 | 338 | 46.86% | 48.75% | 47.29% | 3.14 pp | -40 | 54 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 987 | 470 | 517 | 47.62% | 50.00% | 46.67% | 2.38 pp | -47 | 51 | -0.92 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 810 | 376 | 434 | 46.42% | 44.17% | 45.42% | 3.58 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 810 | 375 | 435 | 46.30% | 39.58% | 46.46% | 3.70 pp | -60 | 47 | -1.28 |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 987 | 457 | 530 | 46.30% | 44.17% | 43.75% | 3.70 pp | -73 | 51 | -1.43 |
| BTC Market Hours | rf | RandomForest | 582 | 251 | 331 | 43.13% | 44.17% | 43.54% | 6.87 pp | -80 | 54 | -1.48 |
| Consolidated Hourly | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 582 | 250 | 332 | 42.96% | 45.42% | 43.33% | 7.04 pp | -82 | 54 | -1.52 |
| BTC Market Hours | lstm | LSTM | 582 | 249 | 333 | 42.78% | 41.67% | 42.92% | 7.22 pp | -84 | 54 | -1.56 |
| Consolidated Hourly | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 636 | 265 | 371 | 41.67% | 42.92% | 40.83% | 8.33 pp | -106 | 54 | -1.96 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 636 | 263 | 373 | 41.35% | 44.58% | 40.62% | 8.65 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 636 | 260 | 376 | 40.88% | 41.25% | 40.42% | 9.12 pp | -116 | 54 | -2.15 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Hourly | rf | RandomForest | 987 | 436 | 551 | 44.17% | 41.67% | 42.92% | 5.83 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 987 | 435 | 552 | 44.07% | 41.67% | 42.08% | 5.93 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Hourly | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 810 | 342 | 468 | 42.22% | 35.42% | 40.62% | 7.78 pp | -126 | 47 | -2.68 |
| BTC Hourly | lstm | LSTM | 987 | 420 | 567 | 42.55% | 37.92% | 40.62% | 7.45 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 810 | 336 | 474 | 41.48% | 36.67% | 41.46% | 8.52 pp | -138 | 47 | -2.94 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 987 | 406 | 581 | 41.13% | 35.00% | 38.75% | 8.87 pp | -175 | 51 | -3.43 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 820 | 320 | 500 | 39.02% | 35.00% | 35.62% | 10.98 pp | -180 | 47 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 987 | 470 | 517 | 47.62% | 50.00% | 46.67% | 2.38 pp | -47 | 51 | -0.92 |
| BTC Hourly | transformer | Transformer | 987 | 457 | 530 | 46.30% | 44.17% | 43.75% | 3.70 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 987 | 436 | 551 | 44.17% | 41.67% | 42.92% | 5.83 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 987 | 435 | 552 | 44.07% | 41.67% | 42.08% | 5.93 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 987 | 420 | 567 | 42.55% | 37.92% | 40.62% | 7.45 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 987 | 406 | 581 | 41.13% | 35.00% | 38.75% | 8.87 pp | -175 | 51 | -3.43 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 810 | 390 | 420 | 48.15% | 46.25% | 47.29% | 1.85 pp | -30 | 47 | -0.64 |
| BTC Daily | nn | NN | 810 | 376 | 434 | 46.42% | 44.17% | 45.42% | 3.58 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 810 | 375 | 435 | 46.30% | 39.58% | 46.46% | 3.70 pp | -60 | 47 | -1.28 |
| BTC Daily | lstm | LSTM | 810 | 342 | 468 | 42.22% | 35.42% | 40.62% | 7.78 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 810 | 336 | 474 | 41.48% | 36.67% | 41.46% | 8.52 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 820 | 320 | 500 | 39.02% | 35.00% | 35.62% | 10.98 pp | -180 | 47 | -3.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 582 | 283 | 299 | 48.63% | 47.08% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 582 | 281 | 301 | 48.28% | 52.92% | 50.00% | 1.72 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 582 | 275 | 307 | 47.25% | 47.50% | 47.08% | 2.75 pp | -32 | 54 | -0.59 |
| BTC Market Hours | rf | RandomForest | 582 | 251 | 331 | 43.13% | 44.17% | 43.54% | 6.87 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 582 | 250 | 332 | 42.96% | 45.42% | 43.33% | 7.04 pp | -82 | 54 | -1.52 |
| BTC Market Hours | lstm | LSTM | 582 | 249 | 333 | 42.78% | 41.67% | 42.92% | 7.22 pp | -84 | 54 | -1.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 636 | 300 | 336 | 47.17% | 48.75% | 48.54% | 2.83 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 636 | 299 | 337 | 47.01% | 50.00% | 47.29% | 2.99 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 636 | 298 | 338 | 46.86% | 48.75% | 47.29% | 3.14 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | rf | RandomForest | 636 | 265 | 371 | 41.67% | 42.92% | 40.83% | 8.33 pp | -106 | 54 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 636 | 263 | 373 | 41.35% | 44.58% | 40.62% | 8.65 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 636 | 260 | 376 | 40.88% | 41.25% | 40.42% | 9.12 pp | -116 | 54 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Hourly | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
