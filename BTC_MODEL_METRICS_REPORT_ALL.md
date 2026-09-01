# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T01:03:10.310396+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1171 | 883 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1047 | 682 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 676 | 444 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 678 | 498 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 17:00:00+00:00 | 95 | 95 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 17:00:00+00:00 | 95 | 95 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 17:00:00+00:00 | 95 | 7 | 88 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 17:00:00+00:00 | 95 | 7 | 88 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 444 | 217 | 227 | 48.87% | 45.42% | 48.87% | 1.13 pp | -10 | 44 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 672 | 329 | 343 | 48.96% | 47.92% | 49.58% | 1.04 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 672 | 325 | 347 | 48.36% | 45.83% | 49.58% | 1.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours | nn | NN | 444 | 210 | 234 | 47.30% | 48.75% | 47.30% | 2.70 pp | -24 | 44 | -0.55 |
| Consolidated Hourly | transformer | Transformer | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| BTC Market Hours | transformer | Transformer | 444 | 203 | 241 | 45.72% | 40.00% | 45.72% | 4.28 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 498 | 229 | 269 | 45.98% | 46.67% | 46.46% | 4.02 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | nn | NN | 498 | 229 | 269 | 45.98% | 44.17% | 46.88% | 4.02 pp | -40 | 43 | -0.93 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 849 | 401 | 448 | 47.23% | 47.08% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 672 | 314 | 358 | 46.73% | 42.92% | 49.17% | 3.27 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 849 | 399 | 450 | 47.00% | 45.00% | 46.67% | 3.00 pp | -51 | 45 | -1.13 |
| BTC Market Hours Daily | transformer | Transformer | 498 | 224 | 274 | 44.98% | 44.58% | 45.00% | 5.02 pp | -50 | 43 | -1.16 |
| Consolidated Hourly | nn | NN | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 9 | -1.22 |
| BTC Market Hours | rf | RandomForest | 444 | 192 | 252 | 43.24% | 43.33% | 43.24% | 6.76 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 444 | 190 | 254 | 42.79% | 40.83% | 42.79% | 7.21 pp | -64 | 44 | -1.45 |
| BTC Hourly | nn | NN | 849 | 383 | 466 | 45.11% | 44.17% | 44.38% | 4.89 pp | -83 | 45 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 498 | 206 | 292 | 41.37% | 41.67% | 41.67% | 8.63 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 444 | 178 | 266 | 40.09% | 38.33% | 40.09% | 9.91 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 849 | 378 | 471 | 44.52% | 43.33% | 43.75% | 5.48 pp | -93 | 45 | -2.07 |
| BTC Daily | lstm | LSTM | 672 | 293 | 379 | 43.60% | 38.33% | 42.92% | 6.40 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 498 | 199 | 299 | 39.96% | 37.92% | 40.62% | 10.04 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 672 | 288 | 384 | 42.86% | 40.83% | 43.75% | 7.14 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 498 | 195 | 303 | 39.16% | 36.25% | 38.96% | 10.84 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 849 | 363 | 486 | 42.76% | 39.17% | 42.29% | 7.24 pp | -123 | 45 | -2.73 |
| BTC Hourly | xgb | XGBoost | 849 | 357 | 492 | 42.05% | 40.00% | 42.29% | 7.95 pp | -135 | 45 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 682 | 271 | 411 | 39.74% | 35.00% | 39.58% | 10.26 pp | -140 | 41 | -3.41 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 849 | 401 | 448 | 47.23% | 47.08% | 47.08% | 2.77 pp | -47 | 45 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 849 | 399 | 450 | 47.00% | 45.00% | 46.67% | 3.00 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 849 | 383 | 466 | 45.11% | 44.17% | 44.38% | 4.89 pp | -83 | 45 | -1.84 |
| BTC Hourly | rf | RandomForest | 849 | 378 | 471 | 44.52% | 43.33% | 43.75% | 5.48 pp | -93 | 45 | -2.07 |
| BTC Hourly | lstm | LSTM | 849 | 363 | 486 | 42.76% | 39.17% | 42.29% | 7.24 pp | -123 | 45 | -2.73 |
| BTC Hourly | xgb | XGBoost | 849 | 357 | 492 | 42.05% | 40.00% | 42.29% | 7.95 pp | -135 | 45 | -3.00 |

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
| Consolidated Hourly | rf | RandomForest | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | transformer | Transformer | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
