# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T10:29:23.236232+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1161 | 873 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1037 | 672 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 653 | 434 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 654 | 487 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 434 | 213 | 221 | 49.08% | 45.42% | 49.08% | 0.92 pp | -8 | 43 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 662 | 324 | 338 | 48.94% | 47.08% | 50.00% | 1.06 pp | -14 | 40 | -0.35 |
| BTC Daily | transformer | Transformer | 662 | 320 | 342 | 48.34% | 45.42% | 49.58% | 1.66 pp | -22 | 40 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 434 | 203 | 231 | 46.77% | 48.75% | 46.77% | 3.23 pp | -28 | 43 | -0.65 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 434 | 199 | 235 | 45.85% | 41.25% | 45.85% | 4.15 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 224 | 263 | 46.00% | 47.08% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Hourly | transformer | Transformer | 839 | 398 | 441 | 47.44% | 47.92% | 46.88% | 2.56 pp | -43 | 45 | -0.96 |
| BTC Daily | nn | NN | 662 | 311 | 351 | 46.98% | 43.33% | 49.58% | 3.02 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | nn | NN | 487 | 221 | 266 | 45.38% | 43.33% | 45.62% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 221 | 266 | 45.38% | 45.42% | 45.42% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 839 | 392 | 447 | 46.72% | 42.92% | 46.46% | 3.28 pp | -55 | 45 | -1.22 |
| BTC Market Hours | lstm | LSTM | 434 | 187 | 247 | 43.09% | 42.92% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 434 | 187 | 247 | 43.09% | 43.33% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 839 | 380 | 459 | 45.29% | 44.17% | 45.00% | 4.71 pp | -79 | 45 | -1.76 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 201 | 286 | 41.27% | 41.67% | 41.46% | 8.73 pp | -85 | 43 | -1.98 |
| BTC Hourly | rf | RandomForest | 839 | 375 | 464 | 44.70% | 43.75% | 44.38% | 5.30 pp | -89 | 45 | -1.98 |
| BTC Daily | lstm | LSTM | 662 | 291 | 371 | 43.96% | 39.58% | 43.54% | 6.04 pp | -80 | 40 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 434 | 174 | 260 | 40.09% | 38.33% | 40.09% | 9.91 pp | -86 | 43 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 196 | 291 | 40.25% | 39.17% | 40.42% | 9.75 pp | -95 | 43 | -2.21 |
| BTC Daily | rf | RandomForest | 662 | 284 | 378 | 42.90% | 41.67% | 44.17% | 7.10 pp | -94 | 40 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 189 | 298 | 38.81% | 35.42% | 38.96% | 11.19 pp | -109 | 43 | -2.53 |
| BTC Hourly | lstm | LSTM | 839 | 361 | 478 | 43.03% | 40.00% | 42.71% | 6.97 pp | -117 | 45 | -2.60 |
| BTC Hourly | xgb | XGBoost | 839 | 355 | 484 | 42.31% | 40.00% | 42.71% | 7.69 pp | -129 | 45 | -2.87 |
| BTC Daily | xgb | XGBoost | 672 | 267 | 405 | 39.73% | 33.75% | 40.00% | 10.27 pp | -138 | 40 | -3.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 839 | 398 | 441 | 47.44% | 47.92% | 46.88% | 2.56 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 839 | 392 | 447 | 46.72% | 42.92% | 46.46% | 3.28 pp | -55 | 45 | -1.22 |
| BTC Hourly | nn | NN | 839 | 380 | 459 | 45.29% | 44.17% | 45.00% | 4.71 pp | -79 | 45 | -1.76 |
| BTC Hourly | rf | RandomForest | 839 | 375 | 464 | 44.70% | 43.75% | 44.38% | 5.30 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 839 | 361 | 478 | 43.03% | 40.00% | 42.71% | 6.97 pp | -117 | 45 | -2.60 |
| BTC Hourly | xgb | XGBoost | 839 | 355 | 484 | 42.31% | 40.00% | 42.71% | 7.69 pp | -129 | 45 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 662 | 324 | 338 | 48.94% | 47.08% | 50.00% | 1.06 pp | -14 | 40 | -0.35 |
| BTC Daily | transformer | Transformer | 662 | 320 | 342 | 48.34% | 45.42% | 49.58% | 1.66 pp | -22 | 40 | -0.55 |
| BTC Daily | nn | NN | 662 | 311 | 351 | 46.98% | 43.33% | 49.58% | 3.02 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 662 | 291 | 371 | 43.96% | 39.58% | 43.54% | 6.04 pp | -80 | 40 | -2.00 |
| BTC Daily | rf | RandomForest | 662 | 284 | 378 | 42.90% | 41.67% | 44.17% | 7.10 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 672 | 267 | 405 | 39.73% | 33.75% | 40.00% | 10.27 pp | -138 | 40 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 434 | 213 | 221 | 49.08% | 45.42% | 49.08% | 0.92 pp | -8 | 43 | -0.19 |
| BTC Market Hours | nn | NN | 434 | 203 | 231 | 46.77% | 48.75% | 46.77% | 3.23 pp | -28 | 43 | -0.65 |
| BTC Market Hours | transformer | Transformer | 434 | 199 | 235 | 45.85% | 41.25% | 45.85% | 4.15 pp | -36 | 43 | -0.84 |
| BTC Market Hours | lstm | LSTM | 434 | 187 | 247 | 43.09% | 42.92% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | rf | RandomForest | 434 | 187 | 247 | 43.09% | 43.33% | 43.09% | 6.91 pp | -60 | 43 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 434 | 174 | 260 | 40.09% | 38.33% | 40.09% | 9.91 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 224 | 263 | 46.00% | 47.08% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | nn | NN | 487 | 221 | 266 | 45.38% | 43.33% | 45.62% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 221 | 266 | 45.38% | 45.42% | 45.42% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 201 | 286 | 41.27% | 41.67% | 41.46% | 8.73 pp | -85 | 43 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 196 | 291 | 40.25% | 39.17% | 40.42% | 9.75 pp | -95 | 43 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 189 | 298 | 38.81% | 35.42% | 38.96% | 11.19 pp | -109 | 43 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
