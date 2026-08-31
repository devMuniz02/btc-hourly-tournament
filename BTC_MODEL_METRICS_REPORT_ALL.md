# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T14:44:52.010865+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1164 | 876 | 288 | 0 |
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
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 325 | 339 | 48.95% | 47.08% | 50.00% | 1.05 pp | -14 | 41 | -0.34 |
| BTC Market Hours | nn | NN | 436 | 205 | 231 | 47.02% | 49.17% | 47.02% | 2.98 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 664 | 319 | 345 | 48.04% | 44.58% | 49.17% | 1.96 pp | -26 | 41 | -0.63 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 490 | 227 | 263 | 46.33% | 47.08% | 46.67% | 3.67 pp | -36 | 43 | -0.84 |
| BTC Market Hours | transformer | Transformer | 436 | 200 | 236 | 45.87% | 40.83% | 45.87% | 4.13 pp | -36 | 43 | -0.84 |
| BTC Daily | nn | NN | 664 | 312 | 352 | 46.99% | 43.33% | 49.58% | 3.01 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 842 | 399 | 443 | 47.39% | 47.92% | 47.08% | 2.61 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 490 | 223 | 267 | 45.51% | 43.33% | 46.04% | 4.49 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 490 | 222 | 268 | 45.31% | 45.00% | 45.42% | 4.69 pp | -46 | 43 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 842 | 395 | 447 | 46.91% | 43.75% | 46.67% | 3.09 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 436 | 188 | 248 | 43.12% | 42.50% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 436 | 188 | 248 | 43.12% | 43.33% | 43.12% | 6.88 pp | -60 | 43 | -1.40 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 842 | 380 | 462 | 45.13% | 43.75% | 44.58% | 4.87 pp | -82 | 45 | -1.82 |
| BTC Daily | lstm | LSTM | 664 | 292 | 372 | 43.98% | 39.58% | 43.33% | 6.02 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 490 | 202 | 288 | 41.22% | 41.67% | 41.25% | 8.78 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 436 | 175 | 261 | 40.14% | 38.33% | 40.14% | 9.86 pp | -86 | 43 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 842 | 375 | 467 | 44.54% | 43.33% | 43.96% | 5.46 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 490 | 196 | 294 | 40.00% | 38.75% | 40.42% | 10.00 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 664 | 284 | 380 | 42.77% | 40.83% | 43.96% | 7.23 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 490 | 191 | 299 | 38.98% | 35.83% | 39.17% | 11.02 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 842 | 361 | 481 | 42.87% | 39.58% | 42.29% | 7.13 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 842 | 355 | 487 | 42.16% | 39.58% | 42.50% | 7.84 pp | -132 | 45 | -2.93 |
| BTC Daily | xgb | XGBoost | 674 | 268 | 406 | 39.76% | 34.17% | 40.00% | 10.24 pp | -138 | 41 | -3.37 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 842 | 399 | 443 | 47.39% | 47.92% | 47.08% | 2.61 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 842 | 395 | 447 | 46.91% | 43.75% | 46.67% | 3.09 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 842 | 380 | 462 | 45.13% | 43.75% | 44.58% | 4.87 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 842 | 375 | 467 | 44.54% | 43.33% | 43.96% | 5.46 pp | -92 | 45 | -2.04 |
| BTC Hourly | lstm | LSTM | 842 | 361 | 481 | 42.87% | 39.58% | 42.29% | 7.13 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 842 | 355 | 487 | 42.16% | 39.58% | 42.50% | 7.84 pp | -132 | 45 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 664 | 325 | 339 | 48.95% | 47.08% | 50.00% | 1.05 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 664 | 319 | 345 | 48.04% | 44.58% | 49.17% | 1.96 pp | -26 | 41 | -0.63 |
| BTC Daily | nn | NN | 664 | 312 | 352 | 46.99% | 43.33% | 49.58% | 3.01 pp | -40 | 41 | -0.98 |
| BTC Daily | lstm | LSTM | 664 | 292 | 372 | 43.98% | 39.58% | 43.33% | 6.02 pp | -80 | 41 | -1.95 |
| BTC Daily | rf | RandomForest | 664 | 284 | 380 | 42.77% | 40.83% | 43.96% | 7.23 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 674 | 268 | 406 | 39.76% | 34.17% | 40.00% | 10.24 pp | -138 | 41 | -3.37 |

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
