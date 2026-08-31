# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T03:04:43.320634+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1156 | 868 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1032 | 667 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 648 | 429 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 650 | 483 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 1 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 23:00:00+00:00 | 83 | 1 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 429 | 211 | 218 | 49.18% | 46.67% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 657 | 321 | 336 | 48.86% | 47.08% | 49.79% | 1.14 pp | -15 | 40 | -0.38 |
| BTC Daily | transformer | Transformer | 657 | 317 | 340 | 48.25% | 45.83% | 49.38% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Market Hours | nn | NN | 429 | 202 | 227 | 47.09% | 50.00% | 47.09% | 2.91 pp | -25 | 42 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 222 | 261 | 45.96% | 47.08% | 46.25% | 4.04 pp | -39 | 42 | -0.93 |
| BTC Market Hours | transformer | Transformer | 429 | 195 | 234 | 45.45% | 40.83% | 45.45% | 4.55 pp | -39 | 42 | -0.93 |
| BTC Hourly | transformer | Transformer | 834 | 396 | 438 | 47.48% | 47.50% | 46.88% | 2.52 pp | -42 | 45 | -0.93 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 657 | 308 | 349 | 46.88% | 42.50% | 49.38% | 3.12 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | nn | NN | 483 | 219 | 264 | 45.34% | 43.75% | 45.62% | 4.66 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 218 | 265 | 45.13% | 44.58% | 45.21% | 4.87 pp | -47 | 42 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 834 | 390 | 444 | 46.76% | 42.50% | 46.46% | 3.24 pp | -54 | 45 | -1.20 |
| BTC Market Hours | lstm | LSTM | 429 | 186 | 243 | 43.36% | 43.33% | 43.36% | 6.64 pp | -57 | 42 | -1.36 |
| BTC Market Hours | rf | RandomForest | 429 | 185 | 244 | 43.12% | 42.92% | 43.12% | 6.88 pp | -59 | 42 | -1.40 |
| BTC Hourly | nn | NN | 834 | 377 | 457 | 45.20% | 43.75% | 44.58% | 4.80 pp | -80 | 45 | -1.78 |
| BTC Daily | lstm | LSTM | 657 | 290 | 367 | 44.14% | 40.00% | 43.54% | 5.86 pp | -77 | 40 | -1.93 |
| BTC Hourly | rf | RandomForest | 834 | 373 | 461 | 44.72% | 43.33% | 44.17% | 5.28 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 199 | 284 | 41.20% | 41.67% | 41.25% | 8.80 pp | -85 | 42 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 429 | 170 | 259 | 39.63% | 37.50% | 39.63% | 10.37 pp | -89 | 42 | -2.12 |
| Consolidated Hourly | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 195 | 288 | 40.37% | 39.17% | 40.42% | 9.63 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 657 | 282 | 375 | 42.92% | 41.25% | 44.17% | 7.08 pp | -93 | 40 | -2.33 |
| BTC Hourly | lstm | LSTM | 834 | 360 | 474 | 43.17% | 40.00% | 42.71% | 6.83 pp | -114 | 45 | -2.53 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 187 | 296 | 38.72% | 35.00% | 38.75% | 11.28 pp | -109 | 42 | -2.60 |
| BTC Hourly | xgb | XGBoost | 834 | 353 | 481 | 42.33% | 39.17% | 42.50% | 7.67 pp | -128 | 45 | -2.84 |
| BTC Daily | xgb | XGBoost | 667 | 265 | 402 | 39.73% | 33.33% | 40.00% | 10.27 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 834 | 396 | 438 | 47.48% | 47.50% | 46.88% | 2.52 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 834 | 390 | 444 | 46.76% | 42.50% | 46.46% | 3.24 pp | -54 | 45 | -1.20 |
| BTC Hourly | nn | NN | 834 | 377 | 457 | 45.20% | 43.75% | 44.58% | 4.80 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 834 | 373 | 461 | 44.72% | 43.33% | 44.17% | 5.28 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 834 | 360 | 474 | 43.17% | 40.00% | 42.71% | 6.83 pp | -114 | 45 | -2.53 |
| BTC Hourly | xgb | XGBoost | 834 | 353 | 481 | 42.33% | 39.17% | 42.50% | 7.67 pp | -128 | 45 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 657 | 321 | 336 | 48.86% | 47.08% | 49.79% | 1.14 pp | -15 | 40 | -0.38 |
| BTC Daily | transformer | Transformer | 657 | 317 | 340 | 48.25% | 45.83% | 49.38% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Daily | nn | NN | 657 | 308 | 349 | 46.88% | 42.50% | 49.38% | 3.12 pp | -41 | 40 | -1.02 |
| BTC Daily | lstm | LSTM | 657 | 290 | 367 | 44.14% | 40.00% | 43.54% | 5.86 pp | -77 | 40 | -1.93 |
| BTC Daily | rf | RandomForest | 657 | 282 | 375 | 42.92% | 41.25% | 44.17% | 7.08 pp | -93 | 40 | -2.33 |
| BTC Daily | xgb | XGBoost | 667 | 265 | 402 | 39.73% | 33.33% | 40.00% | 10.27 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 429 | 211 | 218 | 49.18% | 46.67% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Market Hours | nn | NN | 429 | 202 | 227 | 47.09% | 50.00% | 47.09% | 2.91 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 429 | 195 | 234 | 45.45% | 40.83% | 45.45% | 4.55 pp | -39 | 42 | -0.93 |
| BTC Market Hours | lstm | LSTM | 429 | 186 | 243 | 43.36% | 43.33% | 43.36% | 6.64 pp | -57 | 42 | -1.36 |
| BTC Market Hours | rf | RandomForest | 429 | 185 | 244 | 43.12% | 42.92% | 43.12% | 6.88 pp | -59 | 42 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 429 | 170 | 259 | 39.63% | 37.50% | 39.63% | 10.37 pp | -89 | 42 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 222 | 261 | 45.96% | 47.08% | 46.25% | 4.04 pp | -39 | 42 | -0.93 |
| BTC Market Hours Daily | nn | NN | 483 | 219 | 264 | 45.34% | 43.75% | 45.62% | 4.66 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 218 | 265 | 45.13% | 44.58% | 45.21% | 4.87 pp | -47 | 42 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 199 | 284 | 41.20% | 41.67% | 41.25% | 8.80 pp | -85 | 42 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 195 | 288 | 40.37% | 39.17% | 40.42% | 9.63 pp | -93 | 42 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 187 | 296 | 38.72% | 35.00% | 38.75% | 11.28 pp | -109 | 42 | -2.60 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 8 | -2.12 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
