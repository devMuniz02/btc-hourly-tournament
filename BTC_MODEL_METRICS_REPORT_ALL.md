# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T22:30:59.399524+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 643 | 479 | 162 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 426 | 210 | 216 | 49.30% | 46.67% | 49.30% | 0.70 pp | -6 | 42 | -0.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 412 | 419 | 49.58% | 47.50% | 50.00% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 654 | 318 | 336 | 48.62% | 46.25% | 49.58% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 654 | 317 | 337 | 48.47% | 46.25% | 49.58% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 426 | 201 | 225 | 47.18% | 50.42% | 47.18% | 2.82 pp | -24 | 42 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| BTC Hourly | transformer | Transformer | 831 | 396 | 435 | 47.65% | 46.67% | 45.83% | 2.35 pp | -39 | 45 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 46.67% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 426 | 194 | 232 | 45.54% | 41.25% | 45.54% | 4.46 pp | -38 | 42 | -0.90 |
| BTC Hourly | nn | NN | 831 | 394 | 437 | 47.41% | 46.25% | 46.88% | 2.59 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | nn | NN | 479 | 218 | 261 | 45.51% | 44.17% | 45.51% | 4.49 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 217 | 262 | 45.30% | 45.00% | 45.30% | 4.70 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 654 | 305 | 349 | 46.64% | 41.25% | 48.75% | 3.36 pp | -44 | 40 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| BTC Hourly | rf | RandomForest | 831 | 390 | 441 | 46.93% | 45.42% | 46.04% | 3.07 pp | -51 | 45 | -1.13 |
| BTC Market Hours | lstm | LSTM | 426 | 186 | 240 | 43.66% | 43.75% | 43.66% | 6.34 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 426 | 184 | 242 | 43.19% | 43.33% | 43.19% | 6.81 pp | -58 | 42 | -1.38 |
| BTC Daily | lstm | LSTM | 654 | 289 | 365 | 44.19% | 40.83% | 43.54% | 5.81 pp | -76 | 40 | -1.90 |
| BTC Hourly | lstm | LSTM | 831 | 372 | 459 | 44.77% | 43.33% | 45.62% | 5.23 pp | -87 | 45 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 198 | 281 | 41.34% | 41.67% | 41.34% | 8.66 pp | -83 | 42 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 426 | 169 | 257 | 39.67% | 37.50% | 39.67% | 10.33 pp | -88 | 42 | -2.10 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| BTC Hourly | xgb | XGBoost | 831 | 367 | 464 | 44.16% | 43.33% | 44.17% | 5.84 pp | -97 | 45 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 39.17% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 654 | 279 | 375 | 42.66% | 40.83% | 43.54% | 7.34 pp | -96 | 40 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 185 | 294 | 38.62% | 35.00% | 38.62% | 11.38 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 664 | 262 | 402 | 39.46% | 32.50% | 39.58% | 10.54 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 412 | 419 | 49.58% | 47.50% | 50.00% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Hourly | transformer | Transformer | 831 | 396 | 435 | 47.65% | 46.67% | 45.83% | 2.35 pp | -39 | 45 | -0.87 |
| BTC Hourly | nn | NN | 831 | 394 | 437 | 47.41% | 46.25% | 46.88% | 2.59 pp | -43 | 45 | -0.96 |
| BTC Hourly | rf | RandomForest | 831 | 390 | 441 | 46.93% | 45.42% | 46.04% | 3.07 pp | -51 | 45 | -1.13 |
| BTC Hourly | lstm | LSTM | 831 | 372 | 459 | 44.77% | 43.33% | 45.62% | 5.23 pp | -87 | 45 | -1.93 |
| BTC Hourly | xgb | XGBoost | 831 | 367 | 464 | 44.16% | 43.33% | 44.17% | 5.84 pp | -97 | 45 | -2.16 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 46.67% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | nn | NN | 479 | 218 | 261 | 45.51% | 44.17% | 45.51% | 4.49 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 217 | 262 | 45.30% | 45.00% | 45.30% | 4.70 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 198 | 281 | 41.34% | 41.67% | 41.34% | 8.66 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 39.17% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 185 | 294 | 38.62% | 35.00% | 38.62% | 11.38 pp | -109 | 42 | -2.60 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
