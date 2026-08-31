# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T14:34:39.281088+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1163 | 875 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1039 | 674 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 13:00:00+00:00 | 657 | 436 | 220 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 13:00:00+00:00 | 659 | 490 | 167 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 436 | 215 | 221 | 49.31% | 45.42% | 49.31% | 0.69 pp | -6 | 43 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 326 | 338 | 49.10% | 47.50% | 50.21% | 0.90 pp | -12 | 40 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| BTC Daily | transformer | Transformer | 664 | 320 | 344 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Market Hours | nn | NN | 436 | 205 | 231 | 47.02% | 49.17% | 47.02% | 2.98 pp | -26 | 43 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 490 | 227 | 263 | 46.33% | 47.08% | 46.67% | 3.67 pp | -36 | 43 | -0.84 |
| BTC Market Hours | transformer | Transformer | 436 | 200 | 236 | 45.87% | 40.83% | 45.87% | 4.13 pp | -36 | 43 | -0.84 |
| BTC Daily | nn | NN | 664 | 313 | 351 | 47.14% | 43.75% | 49.79% | 2.86 pp | -38 | 40 | -0.95 |
| BTC Hourly | transformer | Transformer | 841 | 399 | 442 | 47.44% | 47.92% | 47.08% | 2.56 pp | -43 | 45 | -0.96 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 490 | 223 | 267 | 45.51% | 43.33% | 46.04% | 4.49 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 490 | 222 | 268 | 45.31% | 45.00% | 45.42% | 4.69 pp | -46 | 43 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 841 | 395 | 446 | 46.97% | 43.75% | 46.67% | 3.03 pp | -51 | 45 | -1.13 |
| BTC Market Hours | lstm | LSTM | 436 | 188 | 248 | 43.12% | 42.50% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 436 | 188 | 248 | 43.12% | 43.33% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 841 | 380 | 461 | 45.18% | 43.75% | 44.58% | 4.82 pp | -81 | 45 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 490 | 202 | 288 | 41.22% | 41.67% | 41.25% | 8.78 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 436 | 175 | 261 | 40.14% | 38.33% | 40.14% | 9.86 pp | -86 | 43 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 841 | 375 | 466 | 44.59% | 43.33% | 43.96% | 5.41 pp | -91 | 45 | -2.02 |
| BTC Daily | lstm | LSTM | 664 | 291 | 373 | 43.83% | 39.17% | 43.12% | 6.17 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 490 | 196 | 294 | 40.00% | 38.75% | 40.42% | 10.00 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 664 | 285 | 379 | 42.92% | 41.25% | 44.17% | 7.08 pp | -94 | 40 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 490 | 191 | 299 | 38.98% | 35.83% | 39.17% | 11.02 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 841 | 361 | 480 | 42.93% | 39.58% | 42.29% | 7.07 pp | -119 | 45 | -2.64 |
| BTC Hourly | xgb | XGBoost | 841 | 355 | 486 | 42.21% | 39.58% | 42.50% | 7.79 pp | -131 | 45 | -2.91 |
| BTC Daily | xgb | XGBoost | 674 | 269 | 405 | 39.91% | 34.58% | 40.21% | 10.09 pp | -136 | 40 | -3.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 841 | 399 | 442 | 47.44% | 47.92% | 47.08% | 2.56 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 841 | 395 | 446 | 46.97% | 43.75% | 46.67% | 3.03 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 841 | 380 | 461 | 45.18% | 43.75% | 44.58% | 4.82 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 841 | 375 | 466 | 44.59% | 43.33% | 43.96% | 5.41 pp | -91 | 45 | -2.02 |
| BTC Hourly | lstm | LSTM | 841 | 361 | 480 | 42.93% | 39.58% | 42.29% | 7.07 pp | -119 | 45 | -2.64 |
| BTC Hourly | xgb | XGBoost | 841 | 355 | 486 | 42.21% | 39.58% | 42.50% | 7.79 pp | -131 | 45 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 326 | 338 | 49.10% | 47.50% | 50.21% | 0.90 pp | -12 | 40 | -0.30 |
| BTC Daily | transformer | Transformer | 664 | 320 | 344 | 48.19% | 45.00% | 49.38% | 1.81 pp | -24 | 40 | -0.60 |
| BTC Daily | nn | NN | 664 | 313 | 351 | 47.14% | 43.75% | 49.79% | 2.86 pp | -38 | 40 | -0.95 |
| BTC Daily | lstm | LSTM | 664 | 291 | 373 | 43.83% | 39.17% | 43.12% | 6.17 pp | -82 | 40 | -2.05 |
| BTC Daily | rf | RandomForest | 664 | 285 | 379 | 42.92% | 41.25% | 44.17% | 7.08 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 674 | 269 | 405 | 39.91% | 34.58% | 40.21% | 10.09 pp | -136 | 40 | -3.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 436 | 215 | 221 | 49.31% | 45.42% | 49.31% | 0.69 pp | -6 | 43 | -0.14 |
| BTC Market Hours | nn | NN | 436 | 205 | 231 | 47.02% | 49.17% | 47.02% | 2.98 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 436 | 200 | 236 | 45.87% | 40.83% | 45.87% | 4.13 pp | -36 | 43 | -0.84 |
| BTC Market Hours | lstm | LSTM | 436 | 188 | 248 | 43.12% | 42.50% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 436 | 188 | 248 | 43.12% | 43.33% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 436 | 175 | 261 | 40.14% | 38.33% | 40.14% | 9.86 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 490 | 227 | 263 | 46.33% | 47.08% | 46.67% | 3.67 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | nn | NN | 490 | 223 | 267 | 45.51% | 43.33% | 46.04% | 4.49 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 490 | 222 | 268 | 45.31% | 45.00% | 45.42% | 4.69 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 490 | 202 | 288 | 41.22% | 41.67% | 41.25% | 8.78 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 490 | 196 | 294 | 40.00% | 38.75% | 40.42% | 10.00 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 490 | 191 | 299 | 38.98% | 35.83% | 39.17% | 11.02 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
