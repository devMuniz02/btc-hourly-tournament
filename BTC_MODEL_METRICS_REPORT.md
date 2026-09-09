# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T15:33:27.404021+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1310 | 1022 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1186 | 821 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 922 | 583 | 338 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 924 | 637 | 285 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 225 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T22:00:00+00:00 | 226 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 583 | 284 | 299 | 48.71% | 47.50% | 47.92% | 1.29 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 583 | 281 | 302 | 48.20% | 52.50% | 49.79% | 1.80 pp | -21 | 54 | -0.39 |
| BTC Market Hours | transformer | Transformer | 583 | 275 | 308 | 47.17% | 47.08% | 46.88% | 2.83 pp | -33 | 54 | -0.61 |
| Consolidated Hourly | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 14 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 811 | 390 | 421 | 48.09% | 45.83% | 47.08% | 1.91 pp | -31 | 47 | -0.66 |
| BTC Market Hours Daily | nn | NN | 637 | 300 | 337 | 47.10% | 48.33% | 48.33% | 2.90 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 637 | 299 | 338 | 46.94% | 48.75% | 47.50% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 637 | 299 | 338 | 46.94% | 49.58% | 47.29% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 988 | 470 | 518 | 47.57% | 50.00% | 46.46% | 2.43 pp | -48 | 51 | -0.94 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 811 | 376 | 435 | 46.36% | 44.17% | 45.21% | 3.64 pp | -59 | 47 | -1.26 |
| BTC Daily | transformer | Transformer | 811 | 375 | 436 | 46.24% | 39.17% | 46.25% | 3.76 pp | -61 | 47 | -1.30 |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 988 | 458 | 530 | 46.36% | 44.17% | 43.75% | 3.64 pp | -72 | 51 | -1.41 |
| Consolidated Hourly | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 102 | 123 | 45.33% | 45.33% | 45.33% | 4.67 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 583 | 251 | 332 | 43.05% | 43.75% | 43.54% | 6.95 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 583 | 250 | 333 | 42.88% | 45.00% | 43.33% | 7.12 pp | -83 | 54 | -1.54 |
| BTC Market Hours | lstm | LSTM | 583 | 249 | 334 | 42.71% | 41.67% | 42.92% | 7.29 pp | -85 | 54 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 101 | 124 | 44.89% | 44.89% | 44.89% | 5.11 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 637 | 266 | 371 | 41.76% | 42.92% | 41.04% | 8.24 pp | -105 | 54 | -1.94 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 637 | 263 | 374 | 41.29% | 44.17% | 40.62% | 8.71 pp | -111 | 54 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 637 | 261 | 376 | 40.97% | 41.67% | 40.62% | 9.03 pp | -115 | 54 | -2.13 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Hourly | nn | NN | 988 | 436 | 552 | 44.13% | 42.08% | 42.08% | 5.87 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 988 | 436 | 552 | 44.13% | 41.67% | 42.92% | 5.87 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Hourly | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 95 | 130 | 42.22% | 42.22% | 42.22% | 7.78 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 811 | 343 | 468 | 42.29% | 35.83% | 40.83% | 7.71 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| BTC Hourly | lstm | LSTM | 988 | 420 | 568 | 42.51% | 37.50% | 40.42% | 7.49 pp | -148 | 51 | -2.90 |
| BTC Daily | rf | RandomForest | 811 | 336 | 475 | 41.43% | 36.67% | 41.25% | 8.57 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 988 | 406 | 582 | 41.09% | 35.00% | 38.54% | 8.91 pp | -176 | 51 | -3.45 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 821 | 321 | 500 | 39.10% | 35.42% | 35.83% | 10.90 pp | -179 | 47 | -3.81 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 988 | 470 | 518 | 47.57% | 50.00% | 46.46% | 2.43 pp | -48 | 51 | -0.94 |
| BTC Hourly | transformer | Transformer | 988 | 458 | 530 | 46.36% | 44.17% | 43.75% | 3.64 pp | -72 | 51 | -1.41 |
| BTC Hourly | nn | NN | 988 | 436 | 552 | 44.13% | 42.08% | 42.08% | 5.87 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 988 | 436 | 552 | 44.13% | 41.67% | 42.92% | 5.87 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 988 | 420 | 568 | 42.51% | 37.50% | 40.42% | 7.49 pp | -148 | 51 | -2.90 |
| BTC Hourly | xgb | XGBoost | 988 | 406 | 582 | 41.09% | 35.00% | 38.54% | 8.91 pp | -176 | 51 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 811 | 390 | 421 | 48.09% | 45.83% | 47.08% | 1.91 pp | -31 | 47 | -0.66 |
| BTC Daily | nn | NN | 811 | 376 | 435 | 46.36% | 44.17% | 45.21% | 3.64 pp | -59 | 47 | -1.26 |
| BTC Daily | transformer | Transformer | 811 | 375 | 436 | 46.24% | 39.17% | 46.25% | 3.76 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 811 | 343 | 468 | 42.29% | 35.83% | 40.83% | 7.71 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 811 | 336 | 475 | 41.43% | 36.67% | 41.25% | 8.57 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 821 | 321 | 500 | 39.10% | 35.42% | 35.83% | 10.90 pp | -179 | 47 | -3.81 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 583 | 284 | 299 | 48.71% | 47.50% | 47.92% | 1.29 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 583 | 281 | 302 | 48.20% | 52.50% | 49.79% | 1.80 pp | -21 | 54 | -0.39 |
| BTC Market Hours | transformer | Transformer | 583 | 275 | 308 | 47.17% | 47.08% | 46.88% | 2.83 pp | -33 | 54 | -0.61 |
| BTC Market Hours | rf | RandomForest | 583 | 251 | 332 | 43.05% | 43.75% | 43.54% | 6.95 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 583 | 250 | 333 | 42.88% | 45.00% | 43.33% | 7.12 pp | -83 | 54 | -1.54 |
| BTC Market Hours | lstm | LSTM | 583 | 249 | 334 | 42.71% | 41.67% | 42.92% | 7.29 pp | -85 | 54 | -1.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 637 | 300 | 337 | 47.10% | 48.33% | 48.33% | 2.90 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 637 | 299 | 338 | 46.94% | 48.75% | 47.50% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 637 | 299 | 338 | 46.94% | 49.58% | 47.29% | 3.06 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | rf | RandomForest | 637 | 266 | 371 | 41.76% | 42.92% | 41.04% | 8.24 pp | -105 | 54 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 637 | 263 | 374 | 41.29% | 44.17% | 40.62% | 8.71 pp | -111 | 54 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 637 | 261 | 376 | 40.97% | 41.67% | 40.62% | 9.03 pp | -115 | 54 | -2.13 |

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
