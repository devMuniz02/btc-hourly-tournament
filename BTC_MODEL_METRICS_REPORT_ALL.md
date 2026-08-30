# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T22:49:40.860549+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1153 | 865 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1029 | 664 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 642 | 426 | 215 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 644 | 480 | 162 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T09:00:00+00:00 | 80 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T09:00:00+00:00 | 80 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T09:00:00+00:00 | 80 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T09:00:00+00:00 | 81 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 426 | 210 | 216 | 49.30% | 46.67% | 49.30% | 0.70 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 654 | 318 | 336 | 48.62% | 46.25% | 49.58% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 654 | 317 | 337 | 48.47% | 46.25% | 49.58% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 426 | 201 | 225 | 47.18% | 50.42% | 47.18% | 2.82 pp | -24 | 42 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 480 | 222 | 258 | 46.25% | 47.08% | 46.25% | 3.75 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 426 | 194 | 232 | 45.54% | 41.25% | 45.54% | 4.46 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 831 | 395 | 436 | 47.53% | 47.50% | 46.88% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 480 | 219 | 261 | 45.62% | 44.58% | 45.62% | 4.38 pp | -42 | 42 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 480 | 218 | 262 | 45.42% | 45.00% | 45.42% | 4.58 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 654 | 305 | 349 | 46.64% | 41.25% | 48.75% | 3.36 pp | -44 | 40 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 389 | 442 | 46.81% | 42.50% | 46.46% | 3.19 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 426 | 186 | 240 | 43.66% | 43.75% | 43.66% | 6.34 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 426 | 184 | 242 | 43.19% | 43.33% | 43.19% | 6.81 pp | -58 | 42 | -1.38 |
| BTC Hourly | nn | NN | 831 | 375 | 456 | 45.13% | 42.92% | 44.58% | 4.87 pp | -81 | 45 | -1.80 |
| BTC Daily | lstm | LSTM | 654 | 289 | 365 | 44.19% | 40.83% | 43.54% | 5.81 pp | -76 | 40 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 480 | 199 | 281 | 41.46% | 42.08% | 41.46% | 8.54 pp | -82 | 42 | -1.95 |
| BTC Hourly | rf | RandomForest | 831 | 371 | 460 | 44.65% | 42.92% | 44.17% | 5.35 pp | -89 | 45 | -1.98 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 426 | 169 | 257 | 39.67% | 37.50% | 39.67% | 10.33 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 480 | 195 | 285 | 40.62% | 39.58% | 40.62% | 9.38 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 654 | 279 | 375 | 42.66% | 40.83% | 43.54% | 7.34 pp | -96 | 40 | -2.40 |
| BTC Hourly | lstm | LSTM | 831 | 359 | 472 | 43.20% | 40.00% | 42.92% | 6.80 pp | -113 | 45 | -2.51 |
| BTC Market Hours Daily | xgb | XGBoost | 480 | 186 | 294 | 38.75% | 35.00% | 38.75% | 11.25 pp | -108 | 42 | -2.57 |
| BTC Hourly | xgb | XGBoost | 831 | 352 | 479 | 42.36% | 39.17% | 42.71% | 7.64 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 664 | 262 | 402 | 39.46% | 32.50% | 39.58% | 10.54 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 831 | 395 | 436 | 47.53% | 47.50% | 46.88% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 389 | 442 | 46.81% | 42.50% | 46.46% | 3.19 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 831 | 375 | 456 | 45.13% | 42.92% | 44.58% | 4.87 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 831 | 371 | 460 | 44.65% | 42.92% | 44.17% | 5.35 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 831 | 359 | 472 | 43.20% | 40.00% | 42.92% | 6.80 pp | -113 | 45 | -2.51 |
| BTC Hourly | xgb | XGBoost | 831 | 352 | 479 | 42.36% | 39.17% | 42.71% | 7.64 pp | -127 | 45 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 654 | 318 | 336 | 48.62% | 46.25% | 49.58% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 654 | 317 | 337 | 48.47% | 46.25% | 49.58% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 654 | 305 | 349 | 46.64% | 41.25% | 48.75% | 3.36 pp | -44 | 40 | -1.10 |
| BTC Daily | lstm | LSTM | 654 | 289 | 365 | 44.19% | 40.83% | 43.54% | 5.81 pp | -76 | 40 | -1.90 |
| BTC Daily | rf | RandomForest | 654 | 279 | 375 | 42.66% | 40.83% | 43.54% | 7.34 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 664 | 262 | 402 | 39.46% | 32.50% | 39.58% | 10.54 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 426 | 210 | 216 | 49.30% | 46.67% | 49.30% | 0.70 pp | -6 | 42 | -0.14 |
| BTC Market Hours | nn | NN | 426 | 201 | 225 | 47.18% | 50.42% | 47.18% | 2.82 pp | -24 | 42 | -0.57 |
| BTC Market Hours | transformer | Transformer | 426 | 194 | 232 | 45.54% | 41.25% | 45.54% | 4.46 pp | -38 | 42 | -0.90 |
| BTC Market Hours | lstm | LSTM | 426 | 186 | 240 | 43.66% | 43.75% | 43.66% | 6.34 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 426 | 184 | 242 | 43.19% | 43.33% | 43.19% | 6.81 pp | -58 | 42 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 426 | 169 | 257 | 39.67% | 37.50% | 39.67% | 10.33 pp | -88 | 42 | -2.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 480 | 222 | 258 | 46.25% | 47.08% | 46.25% | 3.75 pp | -36 | 42 | -0.86 |
| BTC Market Hours Daily | nn | NN | 480 | 219 | 261 | 45.62% | 44.58% | 45.62% | 4.38 pp | -42 | 42 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 480 | 218 | 262 | 45.42% | 45.00% | 45.42% | 4.58 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 480 | 199 | 281 | 41.46% | 42.08% | 41.46% | 8.54 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 480 | 195 | 285 | 40.62% | 39.58% | 40.62% | 9.38 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 480 | 186 | 294 | 38.75% | 35.00% | 38.75% | 11.25 pp | -108 | 42 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
