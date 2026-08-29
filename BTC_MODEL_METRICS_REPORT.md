# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T22:51:04.295171+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1134 | 846 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1010 | 645 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 610 | 407 | 202 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 612 | 461 | 149 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 14:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 14:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 14:00:00+00:00 | 64 | 1 | 63 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 14:00:00+00:00 | 64 | 1 | 63 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 37 | 27 | 57.81% | 57.81% | 57.81% | 7.81 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 37 | 27 | 57.81% | 57.81% | 57.81% | 7.81 pp | 10 | 7 | 1.43 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 407 | 202 | 205 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| Consolidated Hourly | lstm | LSTM | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 7 | -0.29 |
| BTC Daily | transformer | Transformer | 635 | 311 | 324 | 48.98% | 47.92% | 49.79% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 635 | 309 | 326 | 48.66% | 45.83% | 50.00% | 1.34 pp | -17 | 39 | -0.44 |
| BTC Market Hours | nn | NN | 407 | 193 | 214 | 47.42% | 51.25% | 47.42% | 2.58 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 407 | 188 | 219 | 46.19% | 42.50% | 46.19% | 3.81 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 461 | 213 | 248 | 46.20% | 45.83% | 46.20% | 3.80 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 461 | 212 | 249 | 45.99% | 46.67% | 45.99% | 4.01 pp | -37 | 41 | -0.90 |
| BTC Hourly | transformer | Transformer | 812 | 384 | 428 | 47.29% | 45.83% | 46.46% | 2.71 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 635 | 298 | 337 | 46.93% | 42.50% | 48.96% | 3.07 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 461 | 210 | 251 | 45.55% | 45.00% | 45.55% | 4.45 pp | -41 | 41 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 812 | 382 | 430 | 47.04% | 43.75% | 47.08% | 2.96 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 407 | 180 | 227 | 44.23% | 45.42% | 44.23% | 5.77 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 407 | 175 | 232 | 43.00% | 42.50% | 43.00% | 7.00 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 812 | 366 | 446 | 45.07% | 41.25% | 45.00% | 4.93 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 635 | 281 | 354 | 44.25% | 42.50% | 43.54% | 5.75 pp | -73 | 39 | -1.87 |
| BTC Hourly | rf | RandomForest | 812 | 364 | 448 | 44.83% | 44.58% | 44.79% | 5.17 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 407 | 164 | 243 | 40.29% | 38.75% | 40.29% | 9.71 pp | -79 | 41 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 461 | 189 | 272 | 41.00% | 41.25% | 41.00% | 9.00 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 41 | -2.07 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| BTC Hourly | lstm | LSTM | 812 | 355 | 457 | 43.72% | 42.50% | 44.58% | 6.28 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 635 | 271 | 364 | 42.68% | 42.08% | 43.54% | 7.32 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 461 | 181 | 280 | 39.26% | 36.67% | 39.26% | 10.74 pp | -99 | 41 | -2.41 |
| BTC Hourly | xgb | XGBoost | 812 | 345 | 467 | 42.49% | 40.00% | 42.92% | 7.51 pp | -122 | 44 | -2.77 |
| BTC Daily | xgb | XGBoost | 645 | 252 | 393 | 39.07% | 30.83% | 38.96% | 10.93 pp | -141 | 39 | -3.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 812 | 384 | 428 | 47.29% | 45.83% | 46.46% | 2.71 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 812 | 382 | 430 | 47.04% | 43.75% | 47.08% | 2.96 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 812 | 366 | 446 | 45.07% | 41.25% | 45.00% | 4.93 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 812 | 364 | 448 | 44.83% | 44.58% | 44.79% | 5.17 pp | -84 | 44 | -1.91 |
| BTC Hourly | lstm | LSTM | 812 | 355 | 457 | 43.72% | 42.50% | 44.58% | 6.28 pp | -102 | 44 | -2.32 |
| BTC Hourly | xgb | XGBoost | 812 | 345 | 467 | 42.49% | 40.00% | 42.92% | 7.51 pp | -122 | 44 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 635 | 311 | 324 | 48.98% | 47.92% | 49.79% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 635 | 309 | 326 | 48.66% | 45.83% | 50.00% | 1.34 pp | -17 | 39 | -0.44 |
| BTC Daily | nn | NN | 635 | 298 | 337 | 46.93% | 42.50% | 48.96% | 3.07 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 635 | 281 | 354 | 44.25% | 42.50% | 43.54% | 5.75 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 635 | 271 | 364 | 42.68% | 42.08% | 43.54% | 7.32 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 645 | 252 | 393 | 39.07% | 30.83% | 38.96% | 10.93 pp | -141 | 39 | -3.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 407 | 202 | 205 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 407 | 193 | 214 | 47.42% | 51.25% | 47.42% | 2.58 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 407 | 188 | 219 | 46.19% | 42.50% | 46.19% | 3.81 pp | -31 | 41 | -0.76 |
| BTC Market Hours | lstm | LSTM | 407 | 180 | 227 | 44.23% | 45.42% | 44.23% | 5.77 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 407 | 175 | 232 | 43.00% | 42.50% | 43.00% | 7.00 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 407 | 164 | 243 | 40.29% | 38.75% | 40.29% | 9.71 pp | -79 | 41 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 461 | 213 | 248 | 46.20% | 45.83% | 46.20% | 3.80 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 461 | 212 | 249 | 45.99% | 46.67% | 45.99% | 4.01 pp | -37 | 41 | -0.90 |
| BTC Market Hours Daily | nn | NN | 461 | 210 | 251 | 45.55% | 45.00% | 45.55% | 4.45 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 461 | 189 | 272 | 41.00% | 41.25% | 41.00% | 9.00 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 461 | 181 | 280 | 39.26% | 36.67% | 39.26% | 10.74 pp | -99 | 41 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 37 | 27 | 57.81% | 57.81% | 57.81% | 7.81 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 7 | -0.29 |
| Consolidated Hourly | xgb | XGBoost | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 37 | 27 | 57.81% | 57.81% | 57.81% | 7.81 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
