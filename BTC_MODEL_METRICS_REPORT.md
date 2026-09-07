# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T20:24:46.450868+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1282 | 994 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1157 | 792 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 19:00:00+00:00 | 872 | 554 | 317 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 19:00:00+00:00 | 874 | 608 | 264 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 554 | 270 | 284 | 48.74% | 47.08% | 47.92% | 1.26 pp | -14 | 52 | -0.27 |
| BTC Market Hours | nn | NN | 554 | 265 | 289 | 47.83% | 51.67% | 49.79% | 2.17 pp | -24 | 52 | -0.46 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| BTC Market Hours | transformer | Transformer | 554 | 262 | 292 | 47.29% | 47.50% | 47.71% | 2.71 pp | -30 | 52 | -0.58 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 782 | 376 | 406 | 48.08% | 45.42% | 47.50% | 1.92 pp | -30 | 45 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 608 | 286 | 322 | 47.04% | 50.00% | 47.92% | 2.96 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 608 | 283 | 325 | 46.55% | 47.92% | 47.92% | 3.45 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 608 | 282 | 326 | 46.38% | 48.75% | 47.08% | 3.62 pp | -44 | 52 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 960 | 458 | 502 | 47.71% | 49.58% | 47.29% | 2.29 pp | -44 | 50 | -0.88 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 782 | 362 | 420 | 46.29% | 40.00% | 46.25% | 3.71 pp | -58 | 45 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| BTC Hourly | transformer | Transformer | 960 | 447 | 513 | 46.56% | 45.00% | 43.75% | 3.44 pp | -66 | 50 | -1.32 |
| BTC Daily | nn | NN | 782 | 361 | 421 | 46.16% | 43.75% | 44.79% | 3.84 pp | -60 | 45 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 554 | 240 | 314 | 43.32% | 45.42% | 43.33% | 6.68 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 554 | 238 | 316 | 42.96% | 41.25% | 43.33% | 7.04 pp | -78 | 52 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 554 | 233 | 321 | 42.06% | 44.58% | 42.29% | 7.94 pp | -88 | 52 | -1.69 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 608 | 254 | 354 | 41.78% | 43.75% | 41.04% | 8.22 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 960 | 425 | 535 | 44.27% | 42.08% | 42.50% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | rf | RandomForest | 960 | 425 | 535 | 44.27% | 42.92% | 43.12% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 608 | 246 | 362 | 40.46% | 40.00% | 40.62% | 9.54 pp | -116 | 52 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 608 | 245 | 363 | 40.30% | 41.25% | 39.58% | 9.70 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 782 | 329 | 453 | 42.07% | 34.58% | 39.79% | 7.93 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 960 | 409 | 551 | 42.60% | 37.08% | 41.88% | 7.40 pp | -142 | 50 | -2.84 |
| BTC Daily | rf | RandomForest | 782 | 326 | 456 | 41.69% | 37.50% | 41.67% | 8.31 pp | -130 | 45 | -2.89 |
| BTC Hourly | xgb | XGBoost | 960 | 397 | 563 | 41.35% | 37.50% | 39.38% | 8.65 pp | -166 | 50 | -3.32 |
| BTC Daily | xgb | XGBoost | 792 | 308 | 484 | 38.89% | 35.00% | 36.25% | 11.11 pp | -176 | 45 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 960 | 458 | 502 | 47.71% | 49.58% | 47.29% | 2.29 pp | -44 | 50 | -0.88 |
| BTC Hourly | transformer | Transformer | 960 | 447 | 513 | 46.56% | 45.00% | 43.75% | 3.44 pp | -66 | 50 | -1.32 |
| BTC Hourly | nn | NN | 960 | 425 | 535 | 44.27% | 42.08% | 42.50% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | rf | RandomForest | 960 | 425 | 535 | 44.27% | 42.92% | 43.12% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | lstm | LSTM | 960 | 409 | 551 | 42.60% | 37.08% | 41.88% | 7.40 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 960 | 397 | 563 | 41.35% | 37.50% | 39.38% | 8.65 pp | -166 | 50 | -3.32 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 782 | 376 | 406 | 48.08% | 45.42% | 47.50% | 1.92 pp | -30 | 45 | -0.67 |
| BTC Daily | transformer | Transformer | 782 | 362 | 420 | 46.29% | 40.00% | 46.25% | 3.71 pp | -58 | 45 | -1.29 |
| BTC Daily | nn | NN | 782 | 361 | 421 | 46.16% | 43.75% | 44.79% | 3.84 pp | -60 | 45 | -1.33 |
| BTC Daily | lstm | LSTM | 782 | 329 | 453 | 42.07% | 34.58% | 39.79% | 7.93 pp | -124 | 45 | -2.76 |
| BTC Daily | rf | RandomForest | 782 | 326 | 456 | 41.69% | 37.50% | 41.67% | 8.31 pp | -130 | 45 | -2.89 |
| BTC Daily | xgb | XGBoost | 792 | 308 | 484 | 38.89% | 35.00% | 36.25% | 11.11 pp | -176 | 45 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 554 | 270 | 284 | 48.74% | 47.08% | 47.92% | 1.26 pp | -14 | 52 | -0.27 |
| BTC Market Hours | nn | NN | 554 | 265 | 289 | 47.83% | 51.67% | 49.79% | 2.17 pp | -24 | 52 | -0.46 |
| BTC Market Hours | transformer | Transformer | 554 | 262 | 292 | 47.29% | 47.50% | 47.71% | 2.71 pp | -30 | 52 | -0.58 |
| BTC Market Hours | rf | RandomForest | 554 | 240 | 314 | 43.32% | 45.42% | 43.33% | 6.68 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 554 | 238 | 316 | 42.96% | 41.25% | 43.33% | 7.04 pp | -78 | 52 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 554 | 233 | 321 | 42.06% | 44.58% | 42.29% | 7.94 pp | -88 | 52 | -1.69 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 608 | 286 | 322 | 47.04% | 50.00% | 47.92% | 2.96 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 608 | 283 | 325 | 46.55% | 47.92% | 47.92% | 3.45 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 608 | 282 | 326 | 46.38% | 48.75% | 47.08% | 3.62 pp | -44 | 52 | -0.85 |
| BTC Market Hours Daily | rf | RandomForest | 608 | 254 | 354 | 41.78% | 43.75% | 41.04% | 8.22 pp | -100 | 52 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 608 | 246 | 362 | 40.46% | 40.00% | 40.62% | 9.54 pp | -116 | 52 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 608 | 245 | 363 | 40.30% | 41.25% | 39.58% | 9.70 pp | -118 | 52 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
