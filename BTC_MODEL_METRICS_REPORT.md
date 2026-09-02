# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T06:09:57.162545+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1190 | 902 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1066 | 701 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 708 | 463 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 710 | 517 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 463 | 225 | 238 | 48.60% | 44.17% | 48.60% | 1.40 pp | -13 | 45 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 691 | 338 | 353 | 48.91% | 45.83% | 49.58% | 1.09 pp | -15 | 42 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 463 | 218 | 245 | 47.08% | 48.33% | 47.08% | 2.92 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 691 | 332 | 359 | 48.05% | 46.25% | 49.17% | 1.95 pp | -27 | 42 | -0.64 |
| BTC Market Hours | transformer | Transformer | 463 | 215 | 248 | 46.44% | 40.42% | 46.44% | 3.56 pp | -33 | 45 | -0.73 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 517 | 238 | 279 | 46.03% | 46.67% | 46.46% | 3.97 pp | -41 | 45 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 868 | 411 | 457 | 47.35% | 47.08% | 47.71% | 2.65 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | nn | NN | 517 | 236 | 281 | 45.65% | 43.33% | 46.46% | 4.35 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 517 | 236 | 281 | 45.65% | 47.50% | 46.04% | 4.35 pp | -45 | 45 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 691 | 322 | 369 | 46.60% | 42.92% | 48.96% | 3.40 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 868 | 408 | 460 | 47.00% | 47.50% | 47.08% | 3.00 pp | -52 | 46 | -1.13 |
| BTC Market Hours | rf | RandomForest | 463 | 200 | 263 | 43.20% | 43.33% | 43.20% | 6.80 pp | -63 | 45 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | lstm | LSTM | 463 | 196 | 267 | 42.33% | 40.00% | 42.33% | 7.67 pp | -71 | 45 | -1.58 |
| BTC Hourly | nn | NN | 868 | 391 | 477 | 45.05% | 46.25% | 44.17% | 4.95 pp | -86 | 46 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 45 | -1.89 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 517 | 215 | 302 | 41.59% | 42.08% | 41.67% | 8.41 pp | -87 | 45 | -1.93 |
| BTC Hourly | rf | RandomForest | 868 | 388 | 480 | 44.70% | 45.00% | 44.58% | 5.30 pp | -92 | 46 | -2.00 |
| BTC Daily | lstm | LSTM | 691 | 301 | 390 | 43.56% | 38.75% | 42.50% | 6.44 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 517 | 207 | 310 | 40.04% | 38.33% | 40.62% | 9.96 pp | -103 | 45 | -2.29 |
| BTC Daily | rf | RandomForest | 691 | 297 | 394 | 42.98% | 40.42% | 43.33% | 7.02 pp | -97 | 42 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 517 | 206 | 311 | 39.85% | 37.50% | 39.58% | 10.15 pp | -105 | 45 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 868 | 370 | 498 | 42.63% | 38.75% | 42.29% | 7.37 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 868 | 367 | 501 | 42.28% | 41.25% | 43.33% | 7.72 pp | -134 | 46 | -2.91 |
| BTC Daily | xgb | XGBoost | 701 | 278 | 423 | 39.66% | 35.83% | 39.58% | 10.34 pp | -145 | 42 | -3.45 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 868 | 411 | 457 | 47.35% | 47.08% | 47.71% | 2.65 pp | -46 | 46 | -1.00 |
| BTC Hourly | transformer | Transformer | 868 | 408 | 460 | 47.00% | 47.50% | 47.08% | 3.00 pp | -52 | 46 | -1.13 |
| BTC Hourly | nn | NN | 868 | 391 | 477 | 45.05% | 46.25% | 44.17% | 4.95 pp | -86 | 46 | -1.87 |
| BTC Hourly | rf | RandomForest | 868 | 388 | 480 | 44.70% | 45.00% | 44.58% | 5.30 pp | -92 | 46 | -2.00 |
| BTC Hourly | lstm | LSTM | 868 | 370 | 498 | 42.63% | 38.75% | 42.29% | 7.37 pp | -128 | 46 | -2.78 |
| BTC Hourly | xgb | XGBoost | 868 | 367 | 501 | 42.28% | 41.25% | 43.33% | 7.72 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 691 | 338 | 353 | 48.91% | 45.83% | 49.58% | 1.09 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 691 | 332 | 359 | 48.05% | 46.25% | 49.17% | 1.95 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 691 | 322 | 369 | 46.60% | 42.92% | 48.96% | 3.40 pp | -47 | 42 | -1.12 |
| BTC Daily | lstm | LSTM | 691 | 301 | 390 | 43.56% | 38.75% | 42.50% | 6.44 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 691 | 297 | 394 | 42.98% | 40.42% | 43.33% | 7.02 pp | -97 | 42 | -2.31 |
| BTC Daily | xgb | XGBoost | 701 | 278 | 423 | 39.66% | 35.83% | 39.58% | 10.34 pp | -145 | 42 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 463 | 225 | 238 | 48.60% | 44.17% | 48.60% | 1.40 pp | -13 | 45 | -0.29 |
| BTC Market Hours | nn | NN | 463 | 218 | 245 | 47.08% | 48.33% | 47.08% | 2.92 pp | -27 | 45 | -0.60 |
| BTC Market Hours | transformer | Transformer | 463 | 215 | 248 | 46.44% | 40.42% | 46.44% | 3.56 pp | -33 | 45 | -0.73 |
| BTC Market Hours | rf | RandomForest | 463 | 200 | 263 | 43.20% | 43.33% | 43.20% | 6.80 pp | -63 | 45 | -1.40 |
| BTC Market Hours | lstm | LSTM | 463 | 196 | 267 | 42.33% | 40.00% | 42.33% | 7.67 pp | -71 | 45 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 45 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 517 | 238 | 279 | 46.03% | 46.67% | 46.46% | 3.97 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 517 | 236 | 281 | 45.65% | 43.33% | 46.46% | 4.35 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 517 | 236 | 281 | 45.65% | 47.50% | 46.04% | 4.35 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 517 | 215 | 302 | 41.59% | 42.08% | 41.67% | 8.41 pp | -87 | 45 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 517 | 207 | 310 | 40.04% | 38.33% | 40.62% | 9.96 pp | -103 | 45 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 517 | 206 | 311 | 39.85% | 37.50% | 39.58% | 10.15 pp | -105 | 45 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
