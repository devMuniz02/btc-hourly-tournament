# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T22:33:35.137655+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1185 | 897 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1061 | 696 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 21:00:00+00:00 | 700 | 458 | 241 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 21:00:00+00:00 | 702 | 512 | 188 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 110 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 458 | 222 | 236 | 48.47% | 44.58% | 48.47% | 1.53 pp | -14 | 45 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 686 | 335 | 351 | 48.83% | 46.25% | 49.58% | 1.17 pp | -16 | 41 | -0.39 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 458 | 216 | 242 | 47.16% | 49.17% | 47.16% | 2.84 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 686 | 330 | 356 | 48.10% | 46.25% | 49.38% | 1.90 pp | -26 | 41 | -0.63 |
| Consolidated Hourly | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 458 | 211 | 247 | 46.07% | 40.00% | 46.07% | 3.93 pp | -36 | 45 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 512 | 234 | 278 | 45.70% | 45.83% | 46.04% | 4.30 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 512 | 234 | 278 | 45.70% | 42.92% | 46.46% | 4.30 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 863 | 407 | 456 | 47.16% | 45.83% | 47.08% | 2.84 pp | -49 | 46 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 512 | 232 | 280 | 45.31% | 46.25% | 45.83% | 4.69 pp | -48 | 45 | -1.07 |
| BTC Daily | nn | NN | 686 | 321 | 365 | 46.79% | 43.33% | 49.17% | 3.21 pp | -44 | 41 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| BTC Hourly | transformer | Transformer | 863 | 405 | 458 | 46.93% | 47.08% | 46.67% | 3.07 pp | -53 | 46 | -1.15 |
| Consolidated Hourly | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 458 | 199 | 259 | 43.45% | 43.75% | 43.45% | 6.55 pp | -60 | 45 | -1.33 |
| BTC Market Hours | lstm | LSTM | 458 | 195 | 263 | 42.58% | 40.42% | 42.58% | 7.42 pp | -68 | 45 | -1.51 |
| BTC Hourly | nn | NN | 863 | 389 | 474 | 45.08% | 45.83% | 44.17% | 4.92 pp | -85 | 46 | -1.85 |
| BTC Market Hours | xgb | XGBoost | 458 | 187 | 271 | 40.83% | 39.58% | 40.83% | 9.17 pp | -84 | 45 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 512 | 213 | 299 | 41.60% | 41.67% | 41.88% | 8.40 pp | -86 | 45 | -1.91 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 863 | 383 | 480 | 44.38% | 43.33% | 43.75% | 5.62 pp | -97 | 46 | -2.11 |
| BTC Daily | lstm | LSTM | 686 | 299 | 387 | 43.59% | 38.75% | 42.71% | 6.41 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 512 | 205 | 307 | 40.04% | 37.92% | 40.83% | 9.96 pp | -102 | 45 | -2.27 |
| BTC Daily | rf | RandomForest | 686 | 295 | 391 | 43.00% | 40.83% | 43.54% | 7.00 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 512 | 203 | 309 | 39.65% | 36.67% | 39.38% | 10.35 pp | -106 | 45 | -2.36 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 863 | 367 | 496 | 42.53% | 37.92% | 41.88% | 7.47 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 863 | 364 | 499 | 42.18% | 40.42% | 42.92% | 7.82 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 696 | 276 | 420 | 39.66% | 35.42% | 39.38% | 10.34 pp | -144 | 41 | -3.51 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 863 | 407 | 456 | 47.16% | 45.83% | 47.08% | 2.84 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 863 | 405 | 458 | 46.93% | 47.08% | 46.67% | 3.07 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 863 | 389 | 474 | 45.08% | 45.83% | 44.17% | 4.92 pp | -85 | 46 | -1.85 |
| BTC Hourly | rf | RandomForest | 863 | 383 | 480 | 44.38% | 43.33% | 43.75% | 5.62 pp | -97 | 46 | -2.11 |
| BTC Hourly | lstm | LSTM | 863 | 367 | 496 | 42.53% | 37.92% | 41.88% | 7.47 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 863 | 364 | 499 | 42.18% | 40.42% | 42.92% | 7.82 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 686 | 335 | 351 | 48.83% | 46.25% | 49.58% | 1.17 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 686 | 330 | 356 | 48.10% | 46.25% | 49.38% | 1.90 pp | -26 | 41 | -0.63 |
| BTC Daily | nn | NN | 686 | 321 | 365 | 46.79% | 43.33% | 49.17% | 3.21 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 686 | 299 | 387 | 43.59% | 38.75% | 42.71% | 6.41 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 686 | 295 | 391 | 43.00% | 40.83% | 43.54% | 7.00 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 696 | 276 | 420 | 39.66% | 35.42% | 39.38% | 10.34 pp | -144 | 41 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 458 | 222 | 236 | 48.47% | 44.58% | 48.47% | 1.53 pp | -14 | 45 | -0.31 |
| BTC Market Hours | nn | NN | 458 | 216 | 242 | 47.16% | 49.17% | 47.16% | 2.84 pp | -26 | 45 | -0.58 |
| BTC Market Hours | transformer | Transformer | 458 | 211 | 247 | 46.07% | 40.00% | 46.07% | 3.93 pp | -36 | 45 | -0.80 |
| BTC Market Hours | rf | RandomForest | 458 | 199 | 259 | 43.45% | 43.75% | 43.45% | 6.55 pp | -60 | 45 | -1.33 |
| BTC Market Hours | lstm | LSTM | 458 | 195 | 263 | 42.58% | 40.42% | 42.58% | 7.42 pp | -68 | 45 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 458 | 187 | 271 | 40.83% | 39.58% | 40.83% | 9.17 pp | -84 | 45 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 512 | 234 | 278 | 45.70% | 45.83% | 46.04% | 4.30 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 512 | 234 | 278 | 45.70% | 42.92% | 46.46% | 4.30 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 512 | 232 | 280 | 45.31% | 46.25% | 45.83% | 4.69 pp | -48 | 45 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 512 | 213 | 299 | 41.60% | 41.67% | 41.88% | 8.40 pp | -86 | 45 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 512 | 205 | 307 | 40.04% | 37.92% | 40.83% | 9.96 pp | -102 | 45 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 512 | 203 | 309 | 39.65% | 36.67% | 39.38% | 10.35 pp | -106 | 45 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
