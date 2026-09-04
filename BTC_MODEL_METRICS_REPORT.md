# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T20:23:12.464087+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1232 | 944 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1108 | 743 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 19:00:00+00:00 | 784 | 505 | 278 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 19:00:00+00:00 | 786 | 559 | 225 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 153 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T22:00:00+00:00 | 154 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 505 | 243 | 262 | 48.12% | 44.58% | 48.12% | 1.88 pp | -19 | 48 | -0.40 |
| BTC Market Hours | transformer | Transformer | 505 | 241 | 264 | 47.72% | 46.25% | 48.33% | 2.28 pp | -23 | 48 | -0.48 |
| BTC Daily | mlp_sklearn | MLPClassifier | 733 | 355 | 378 | 48.43% | 47.08% | 48.33% | 1.57 pp | -23 | 43 | -0.53 |
| BTC Market Hours | nn | NN | 505 | 239 | 266 | 47.33% | 50.42% | 48.33% | 2.67 pp | -27 | 48 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 559 | 264 | 295 | 47.23% | 50.00% | 48.12% | 2.77 pp | -31 | 48 | -0.65 |
| BTC Daily | transformer | Transformer | 733 | 350 | 383 | 47.75% | 46.67% | 49.58% | 2.25 pp | -33 | 43 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 910 | 436 | 474 | 47.91% | 50.83% | 48.12% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | nn | NN | 559 | 260 | 299 | 46.51% | 45.83% | 47.92% | 3.49 pp | -39 | 48 | -0.81 |
| Consolidated Hourly | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 559 | 259 | 300 | 46.33% | 49.58% | 46.88% | 3.67 pp | -41 | 48 | -0.85 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 910 | 430 | 480 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 733 | 339 | 394 | 46.25% | 44.58% | 47.08% | 3.75 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 505 | 218 | 287 | 43.17% | 41.67% | 43.33% | 6.83 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 505 | 216 | 289 | 42.77% | 43.75% | 42.92% | 7.23 pp | -73 | 48 | -1.52 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 505 | 208 | 297 | 41.19% | 42.50% | 41.67% | 8.81 pp | -89 | 48 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 559 | 231 | 328 | 41.32% | 41.67% | 40.42% | 8.68 pp | -97 | 48 | -2.02 |
| BTC Hourly | nn | NN | 910 | 405 | 505 | 44.51% | 44.17% | 42.29% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 910 | 405 | 505 | 44.51% | 44.58% | 44.17% | 5.49 pp | -100 | 48 | -2.08 |
| Consolidated Hourly | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 559 | 225 | 334 | 40.25% | 38.33% | 40.42% | 9.75 pp | -109 | 48 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 559 | 224 | 335 | 40.07% | 40.83% | 39.17% | 9.93 pp | -111 | 48 | -2.31 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 3 | -2.33 |
| BTC Daily | lstm | LSTM | 733 | 315 | 418 | 42.97% | 36.67% | 41.25% | 7.03 pp | -103 | 43 | -2.40 |
| BTC Daily | rf | RandomForest | 733 | 312 | 421 | 42.56% | 40.42% | 43.33% | 7.44 pp | -109 | 43 | -2.53 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 910 | 389 | 521 | 42.75% | 39.58% | 41.67% | 7.25 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 910 | 382 | 528 | 41.98% | 41.25% | 40.83% | 8.02 pp | -146 | 48 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 743 | 293 | 450 | 39.43% | 35.83% | 38.12% | 10.57 pp | -157 | 43 | -3.65 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 910 | 436 | 474 | 47.91% | 50.83% | 48.12% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 910 | 430 | 480 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 910 | 405 | 505 | 44.51% | 44.17% | 42.29% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 910 | 405 | 505 | 44.51% | 44.58% | 44.17% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | lstm | LSTM | 910 | 389 | 521 | 42.75% | 39.58% | 41.67% | 7.25 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 910 | 382 | 528 | 41.98% | 41.25% | 40.83% | 8.02 pp | -146 | 48 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 733 | 355 | 378 | 48.43% | 47.08% | 48.33% | 1.57 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 733 | 350 | 383 | 47.75% | 46.67% | 49.58% | 2.25 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 733 | 339 | 394 | 46.25% | 44.58% | 47.08% | 3.75 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 733 | 315 | 418 | 42.97% | 36.67% | 41.25% | 7.03 pp | -103 | 43 | -2.40 |
| BTC Daily | rf | RandomForest | 733 | 312 | 421 | 42.56% | 40.42% | 43.33% | 7.44 pp | -109 | 43 | -2.53 |
| BTC Daily | xgb | XGBoost | 743 | 293 | 450 | 39.43% | 35.83% | 38.12% | 10.57 pp | -157 | 43 | -3.65 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 505 | 243 | 262 | 48.12% | 44.58% | 48.12% | 1.88 pp | -19 | 48 | -0.40 |
| BTC Market Hours | transformer | Transformer | 505 | 241 | 264 | 47.72% | 46.25% | 48.33% | 2.28 pp | -23 | 48 | -0.48 |
| BTC Market Hours | nn | NN | 505 | 239 | 266 | 47.33% | 50.42% | 48.33% | 2.67 pp | -27 | 48 | -0.56 |
| BTC Market Hours | lstm | LSTM | 505 | 218 | 287 | 43.17% | 41.67% | 43.33% | 6.83 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 505 | 216 | 289 | 42.77% | 43.75% | 42.92% | 7.23 pp | -73 | 48 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 505 | 208 | 297 | 41.19% | 42.50% | 41.67% | 8.81 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 559 | 264 | 295 | 47.23% | 50.00% | 48.12% | 2.77 pp | -31 | 48 | -0.65 |
| BTC Market Hours Daily | nn | NN | 559 | 260 | 299 | 46.51% | 45.83% | 47.92% | 3.49 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 559 | 259 | 300 | 46.33% | 49.58% | 46.88% | 3.67 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 559 | 231 | 328 | 41.32% | 41.67% | 40.42% | 8.68 pp | -97 | 48 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 559 | 225 | 334 | 40.25% | 38.33% | 40.42% | 9.75 pp | -109 | 48 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 559 | 224 | 335 | 40.07% | 40.83% | 39.17% | 9.93 pp | -111 | 48 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 65 | 88 | 42.48% | 42.48% | 42.48% | 7.52 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 39 | 15 | 24 | 38.46% | 38.46% | 38.46% | 11.54 pp | -9 | 3 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
