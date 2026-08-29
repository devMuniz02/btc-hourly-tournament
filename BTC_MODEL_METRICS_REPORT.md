# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T06:57:33.574197+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1122 | 834 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 998 | 633 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 588 | 395 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 590 | 449 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T04:00:00+00:00 | 53 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T04:00:00+00:00 | 53 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T04:00:00+00:00 | 53 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T04:00:00+00:00 | 54 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 395 | 194 | 201 | 49.11% | 47.50% | 49.11% | 0.89 pp | -7 | 40 | -0.17 |
| BTC Daily | transformer | Transformer | 623 | 306 | 317 | 49.12% | 47.92% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 623 | 305 | 318 | 48.96% | 47.08% | 50.42% | 1.04 pp | -13 | 39 | -0.33 |
| BTC Market Hours | nn | NN | 395 | 185 | 210 | 46.84% | 49.17% | 46.84% | 3.16 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 395 | 184 | 211 | 46.58% | 43.33% | 46.58% | 3.42 pp | -27 | 40 | -0.68 |
| BTC Market Hours Daily | transformer | Transformer | 449 | 207 | 242 | 46.10% | 47.92% | 46.10% | 3.90 pp | -35 | 40 | -0.88 |
| BTC Daily | nn | NN | 623 | 294 | 329 | 47.19% | 44.17% | 49.38% | 2.81 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 449 | 206 | 243 | 45.88% | 45.00% | 45.88% | 4.12 pp | -37 | 40 | -0.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 800 | 377 | 423 | 47.12% | 44.58% | 47.08% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 800 | 377 | 423 | 47.12% | 44.58% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 449 | 203 | 246 | 45.21% | 45.00% | 45.21% | 4.79 pp | -43 | 40 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| BTC Market Hours | lstm | LSTM | 395 | 172 | 223 | 43.54% | 43.33% | 43.54% | 6.46 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 395 | 168 | 227 | 42.53% | 41.25% | 42.53% | 7.47 pp | -59 | 40 | -1.48 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 623 | 277 | 346 | 44.46% | 42.92% | 44.17% | 5.54 pp | -69 | 39 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 395 | 161 | 234 | 40.76% | 39.58% | 40.76% | 9.24 pp | -73 | 40 | -1.82 |
| BTC Hourly | nn | NN | 800 | 359 | 441 | 44.88% | 40.83% | 45.21% | 5.13 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 800 | 356 | 444 | 44.50% | 43.33% | 44.38% | 5.50 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 449 | 182 | 267 | 40.53% | 39.17% | 40.53% | 9.47 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 449 | 181 | 268 | 40.31% | 38.75% | 40.31% | 9.69 pp | -87 | 40 | -2.17 |
| BTC Hourly | lstm | LSTM | 800 | 352 | 448 | 44.00% | 43.75% | 45.42% | 6.00 pp | -96 | 43 | -2.23 |
| BTC Daily | rf | RandomForest | 623 | 267 | 356 | 42.86% | 42.50% | 43.75% | 7.14 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 449 | 176 | 273 | 39.20% | 37.08% | 39.20% | 10.80 pp | -97 | 40 | -2.42 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| BTC Hourly | xgb | XGBoost | 800 | 340 | 460 | 42.50% | 39.58% | 43.96% | 7.50 pp | -120 | 43 | -2.79 |
| BTC Daily | xgb | XGBoost | 633 | 250 | 383 | 39.49% | 32.50% | 40.00% | 10.51 pp | -133 | 39 | -3.41 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 800 | 377 | 423 | 47.12% | 44.58% | 47.08% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 800 | 377 | 423 | 47.12% | 44.58% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 800 | 359 | 441 | 44.88% | 40.83% | 45.21% | 5.13 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 800 | 356 | 444 | 44.50% | 43.33% | 44.38% | 5.50 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 800 | 352 | 448 | 44.00% | 43.75% | 45.42% | 6.00 pp | -96 | 43 | -2.23 |
| BTC Hourly | xgb | XGBoost | 800 | 340 | 460 | 42.50% | 39.58% | 43.96% | 7.50 pp | -120 | 43 | -2.79 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 623 | 306 | 317 | 49.12% | 47.92% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 623 | 305 | 318 | 48.96% | 47.08% | 50.42% | 1.04 pp | -13 | 39 | -0.33 |
| BTC Daily | nn | NN | 623 | 294 | 329 | 47.19% | 44.17% | 49.38% | 2.81 pp | -35 | 39 | -0.90 |
| BTC Daily | lstm | LSTM | 623 | 277 | 346 | 44.46% | 42.92% | 44.17% | 5.54 pp | -69 | 39 | -1.77 |
| BTC Daily | rf | RandomForest | 623 | 267 | 356 | 42.86% | 42.50% | 43.75% | 7.14 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 633 | 250 | 383 | 39.49% | 32.50% | 40.00% | 10.51 pp | -133 | 39 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 395 | 194 | 201 | 49.11% | 47.50% | 49.11% | 0.89 pp | -7 | 40 | -0.17 |
| BTC Market Hours | nn | NN | 395 | 185 | 210 | 46.84% | 49.17% | 46.84% | 3.16 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 395 | 184 | 211 | 46.58% | 43.33% | 46.58% | 3.42 pp | -27 | 40 | -0.68 |
| BTC Market Hours | lstm | LSTM | 395 | 172 | 223 | 43.54% | 43.33% | 43.54% | 6.46 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 395 | 168 | 227 | 42.53% | 41.25% | 42.53% | 7.47 pp | -59 | 40 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 395 | 161 | 234 | 40.76% | 39.58% | 40.76% | 9.24 pp | -73 | 40 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 449 | 207 | 242 | 46.10% | 47.92% | 46.10% | 3.90 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 449 | 206 | 243 | 45.88% | 45.00% | 45.88% | 4.12 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 449 | 203 | 246 | 45.21% | 45.00% | 45.21% | 4.79 pp | -43 | 40 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 449 | 182 | 267 | 40.53% | 39.17% | 40.53% | 9.47 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 449 | 181 | 268 | 40.31% | 38.75% | 40.31% | 9.69 pp | -87 | 40 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 449 | 176 | 273 | 39.20% | 37.08% | 39.20% | 10.80 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
