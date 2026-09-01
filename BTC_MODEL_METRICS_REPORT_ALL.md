# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T02:12:46.673113+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1172 | 884 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1047 | 682 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 676 | 444 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 678 | 498 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 444 | 217 | 227 | 48.87% | 45.42% | 48.87% | 1.13 pp | -10 | 44 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 672 | 329 | 343 | 48.96% | 47.92% | 49.58% | 1.04 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 672 | 325 | 347 | 48.36% | 45.83% | 49.58% | 1.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours | nn | NN | 444 | 210 | 234 | 47.30% | 48.75% | 47.30% | 2.70 pp | -24 | 44 | -0.55 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| BTC Market Hours | transformer | Transformer | 444 | 203 | 241 | 45.72% | 40.00% | 45.72% | 4.28 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 498 | 229 | 269 | 45.98% | 46.67% | 46.46% | 4.02 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | nn | NN | 498 | 229 | 269 | 45.98% | 44.17% | 46.88% | 4.02 pp | -40 | 43 | -0.93 |
| BTC Hourly | transformer | Transformer | 850 | 401 | 449 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Daily | nn | NN | 672 | 314 | 358 | 46.73% | 42.92% | 49.17% | 3.27 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 850 | 399 | 451 | 46.94% | 44.58% | 46.46% | 3.06 pp | -52 | 45 | -1.16 |
| BTC Market Hours Daily | transformer | Transformer | 498 | 224 | 274 | 44.98% | 44.58% | 45.00% | 5.02 pp | -50 | 43 | -1.16 |
| BTC Market Hours | rf | RandomForest | 444 | 192 | 252 | 43.24% | 43.33% | 43.24% | 6.76 pp | -60 | 44 | -1.36 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 444 | 190 | 254 | 42.79% | 40.83% | 42.79% | 7.21 pp | -64 | 44 | -1.45 |
| BTC Hourly | nn | NN | 850 | 384 | 466 | 45.18% | 44.58% | 44.58% | 4.82 pp | -82 | 45 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 498 | 206 | 292 | 41.37% | 41.67% | 41.67% | 8.63 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 444 | 178 | 266 | 40.09% | 38.33% | 40.09% | 9.91 pp | -88 | 44 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 850 | 378 | 472 | 44.47% | 43.33% | 43.75% | 5.53 pp | -94 | 45 | -2.09 |
| BTC Daily | lstm | LSTM | 672 | 293 | 379 | 43.60% | 38.33% | 42.92% | 6.40 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 498 | 199 | 299 | 39.96% | 37.92% | 40.62% | 10.04 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 672 | 288 | 384 | 42.86% | 40.83% | 43.75% | 7.14 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 498 | 195 | 303 | 39.16% | 36.25% | 38.96% | 10.84 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 850 | 363 | 487 | 42.71% | 38.75% | 42.08% | 7.29 pp | -124 | 45 | -2.76 |
| BTC Hourly | xgb | XGBoost | 850 | 357 | 493 | 42.00% | 40.00% | 42.29% | 8.00 pp | -136 | 45 | -3.02 |
| BTC Daily | xgb | XGBoost | 682 | 271 | 411 | 39.74% | 35.00% | 39.58% | 10.26 pp | -140 | 41 | -3.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 850 | 401 | 449 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 850 | 399 | 451 | 46.94% | 44.58% | 46.46% | 3.06 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 850 | 384 | 466 | 45.18% | 44.58% | 44.58% | 4.82 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 850 | 378 | 472 | 44.47% | 43.33% | 43.75% | 5.53 pp | -94 | 45 | -2.09 |
| BTC Hourly | lstm | LSTM | 850 | 363 | 487 | 42.71% | 38.75% | 42.08% | 7.29 pp | -124 | 45 | -2.76 |
| BTC Hourly | xgb | XGBoost | 850 | 357 | 493 | 42.00% | 40.00% | 42.29% | 8.00 pp | -136 | 45 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 672 | 329 | 343 | 48.96% | 47.92% | 49.58% | 1.04 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 672 | 325 | 347 | 48.36% | 45.83% | 49.58% | 1.64 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 672 | 314 | 358 | 46.73% | 42.92% | 49.17% | 3.27 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 672 | 293 | 379 | 43.60% | 38.33% | 42.92% | 6.40 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 672 | 288 | 384 | 42.86% | 40.83% | 43.75% | 7.14 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 682 | 271 | 411 | 39.74% | 35.00% | 39.58% | 10.26 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 444 | 217 | 227 | 48.87% | 45.42% | 48.87% | 1.13 pp | -10 | 44 | -0.23 |
| BTC Market Hours | nn | NN | 444 | 210 | 234 | 47.30% | 48.75% | 47.30% | 2.70 pp | -24 | 44 | -0.55 |
| BTC Market Hours | transformer | Transformer | 444 | 203 | 241 | 45.72% | 40.00% | 45.72% | 4.28 pp | -38 | 44 | -0.86 |
| BTC Market Hours | rf | RandomForest | 444 | 192 | 252 | 43.24% | 43.33% | 43.24% | 6.76 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 444 | 190 | 254 | 42.79% | 40.83% | 42.79% | 7.21 pp | -64 | 44 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 444 | 178 | 266 | 40.09% | 38.33% | 40.09% | 9.91 pp | -88 | 44 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 498 | 229 | 269 | 45.98% | 46.67% | 46.46% | 4.02 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | nn | NN | 498 | 229 | 269 | 45.98% | 44.17% | 46.88% | 4.02 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 498 | 224 | 274 | 44.98% | 44.58% | 45.00% | 5.02 pp | -50 | 43 | -1.16 |
| BTC Market Hours Daily | rf | RandomForest | 498 | 206 | 292 | 41.37% | 41.67% | 41.67% | 8.63 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 498 | 199 | 299 | 39.96% | 37.92% | 40.62% | 10.04 pp | -100 | 43 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 498 | 195 | 303 | 39.16% | 36.25% | 38.96% | 10.84 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
