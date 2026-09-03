# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T19:56:41.946334+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1216 | 928 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1092 | 727 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 18:00:00+00:00 | 754 | 489 | 264 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 18:00:00+00:00 | 756 | 543 | 211 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 489 | 235 | 254 | 48.06% | 43.75% | 47.92% | 1.94 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 717 | 347 | 370 | 48.40% | 46.67% | 48.54% | 1.60 pp | -23 | 43 | -0.53 |
| BTC Market Hours | nn | NN | 489 | 229 | 260 | 46.83% | 48.75% | 47.29% | 3.17 pp | -31 | 47 | -0.66 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 489 | 228 | 261 | 46.63% | 42.50% | 47.08% | 3.37 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 717 | 341 | 376 | 47.56% | 45.42% | 49.79% | 2.44 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 543 | 252 | 291 | 46.41% | 49.58% | 47.50% | 3.59 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 894 | 426 | 468 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 543 | 249 | 294 | 45.86% | 47.92% | 47.08% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 543 | 249 | 294 | 45.86% | 44.58% | 46.88% | 4.14 pp | -45 | 47 | -0.96 |
| Consolidated Hourly | lstm | LSTM | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 894 | 423 | 471 | 47.32% | 48.33% | 47.08% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Daily | nn | NN | 717 | 332 | 385 | 46.30% | 43.75% | 47.92% | 3.70 pp | -53 | 43 | -1.23 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 489 | 211 | 278 | 43.15% | 40.83% | 43.12% | 6.85 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 489 | 210 | 279 | 42.94% | 42.50% | 43.12% | 7.06 pp | -69 | 47 | -1.47 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 489 | 199 | 290 | 40.70% | 40.00% | 40.62% | 9.30 pp | -91 | 47 | -1.94 |
| BTC Market Hours Daily | rf | RandomForest | 543 | 224 | 319 | 41.25% | 41.67% | 41.25% | 8.75 pp | -95 | 47 | -2.02 |
| BTC Hourly | nn | NN | 894 | 399 | 495 | 44.63% | 45.00% | 42.50% | 5.37 pp | -96 | 47 | -2.04 |
| BTC Hourly | rf | RandomForest | 894 | 399 | 495 | 44.63% | 45.83% | 44.17% | 5.37 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 543 | 218 | 325 | 40.15% | 37.92% | 40.62% | 9.85 pp | -107 | 47 | -2.28 |
| BTC Daily | lstm | LSTM | 717 | 309 | 408 | 43.10% | 37.50% | 42.08% | 6.90 pp | -99 | 43 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 543 | 216 | 327 | 39.78% | 39.58% | 39.38% | 10.22 pp | -111 | 47 | -2.36 |
| BTC Daily | rf | RandomForest | 717 | 306 | 411 | 42.68% | 40.83% | 43.54% | 7.32 pp | -105 | 43 | -2.44 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 894 | 383 | 511 | 42.84% | 39.58% | 42.29% | 7.16 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 894 | 377 | 517 | 42.17% | 42.92% | 42.08% | 7.83 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 727 | 287 | 440 | 39.48% | 35.00% | 38.54% | 10.52 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 894 | 426 | 468 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 894 | 423 | 471 | 47.32% | 48.33% | 47.08% | 2.68 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 894 | 399 | 495 | 44.63% | 45.00% | 42.50% | 5.37 pp | -96 | 47 | -2.04 |
| BTC Hourly | rf | RandomForest | 894 | 399 | 495 | 44.63% | 45.83% | 44.17% | 5.37 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 894 | 383 | 511 | 42.84% | 39.58% | 42.29% | 7.16 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 894 | 377 | 517 | 42.17% | 42.92% | 42.08% | 7.83 pp | -140 | 47 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 717 | 347 | 370 | 48.40% | 46.67% | 48.54% | 1.60 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 717 | 341 | 376 | 47.56% | 45.42% | 49.79% | 2.44 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 717 | 332 | 385 | 46.30% | 43.75% | 47.92% | 3.70 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 717 | 309 | 408 | 43.10% | 37.50% | 42.08% | 6.90 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 717 | 306 | 411 | 42.68% | 40.83% | 43.54% | 7.32 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 727 | 287 | 440 | 39.48% | 35.00% | 38.54% | 10.52 pp | -153 | 43 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 489 | 235 | 254 | 48.06% | 43.75% | 47.92% | 1.94 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 489 | 229 | 260 | 46.83% | 48.75% | 47.29% | 3.17 pp | -31 | 47 | -0.66 |
| BTC Market Hours | transformer | Transformer | 489 | 228 | 261 | 46.63% | 42.50% | 47.08% | 3.37 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 489 | 211 | 278 | 43.15% | 40.83% | 43.12% | 6.85 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 489 | 210 | 279 | 42.94% | 42.50% | 43.12% | 7.06 pp | -69 | 47 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 489 | 199 | 290 | 40.70% | 40.00% | 40.62% | 9.30 pp | -91 | 47 | -1.94 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 543 | 252 | 291 | 46.41% | 49.58% | 47.50% | 3.59 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 543 | 249 | 294 | 45.86% | 47.92% | 47.08% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | nn | NN | 543 | 249 | 294 | 45.86% | 44.58% | 46.88% | 4.14 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 543 | 224 | 319 | 41.25% | 41.67% | 41.25% | 8.75 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 543 | 218 | 325 | 40.15% | 37.92% | 40.62% | 9.85 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 543 | 216 | 327 | 39.78% | 39.58% | 39.38% | 10.22 pp | -111 | 47 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
