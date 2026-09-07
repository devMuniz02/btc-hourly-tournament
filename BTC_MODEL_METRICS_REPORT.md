# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T00:07:47.460970+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1268 | 980 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1143 | 778 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 849 | 540 | 308 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 23:00:00+00:00 | 851 | 594 | 255 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 185 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 14:00:00+00:00 | 185 | 56 | 129 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 540 | 262 | 278 | 48.52% | 46.25% | 48.12% | 1.48 pp | -16 | 51 | -0.31 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 540 | 258 | 282 | 47.78% | 48.75% | 48.12% | 2.22 pp | -24 | 51 | -0.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 768 | 371 | 397 | 48.31% | 46.67% | 48.33% | 1.69 pp | -26 | 45 | -0.58 |
| BTC Market Hours Daily | transformer | Transformer | 594 | 282 | 312 | 47.47% | 50.83% | 48.75% | 2.53 pp | -30 | 51 | -0.59 |
| BTC Market Hours | nn | NN | 540 | 255 | 285 | 47.22% | 50.00% | 48.96% | 2.78 pp | -30 | 51 | -0.59 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 946 | 453 | 493 | 47.89% | 50.42% | 47.50% | 2.11 pp | -40 | 49 | -0.82 |
| BTC Market Hours Daily | nn | NN | 594 | 276 | 318 | 46.46% | 46.25% | 47.92% | 3.54 pp | -42 | 51 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 594 | 275 | 319 | 46.30% | 50.42% | 47.08% | 3.70 pp | -44 | 51 | -0.86 |
| BTC Daily | transformer | Transformer | 768 | 359 | 409 | 46.74% | 41.67% | 47.08% | 3.26 pp | -50 | 45 | -1.11 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 768 | 356 | 412 | 46.35% | 45.00% | 46.04% | 3.65 pp | -56 | 45 | -1.24 |
| BTC Hourly | transformer | Transformer | 946 | 442 | 504 | 46.72% | 46.25% | 45.00% | 3.28 pp | -62 | 49 | -1.27 |
| BTC Market Hours | rf | RandomForest | 540 | 233 | 307 | 43.15% | 45.00% | 43.33% | 6.85 pp | -74 | 51 | -1.45 |
| BTC Market Hours | lstm | LSTM | 540 | 232 | 308 | 42.96% | 41.67% | 43.96% | 7.04 pp | -76 | 51 | -1.49 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 540 | 224 | 316 | 41.48% | 42.92% | 41.88% | 8.52 pp | -92 | 51 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 594 | 248 | 346 | 41.75% | 45.00% | 41.46% | 8.25 pp | -98 | 51 | -1.92 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 946 | 420 | 526 | 44.40% | 42.50% | 42.92% | 5.60 pp | -106 | 49 | -2.16 |
| BTC Hourly | rf | RandomForest | 946 | 419 | 527 | 44.29% | 44.17% | 43.75% | 5.71 pp | -108 | 49 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 594 | 240 | 354 | 40.40% | 38.75% | 40.00% | 9.60 pp | -114 | 51 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 594 | 237 | 357 | 39.90% | 41.25% | 39.17% | 10.10 pp | -120 | 51 | -2.35 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 768 | 325 | 443 | 42.32% | 36.25% | 40.42% | 7.68 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 768 | 322 | 446 | 41.93% | 38.75% | 42.29% | 8.07 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 946 | 405 | 541 | 42.81% | 37.08% | 42.29% | 7.19 pp | -136 | 49 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | xgb | XGBoost | 946 | 396 | 550 | 41.86% | 40.42% | 40.62% | 8.14 pp | -154 | 49 | -3.14 |
| BTC Daily | xgb | XGBoost | 778 | 304 | 474 | 39.07% | 35.00% | 36.25% | 10.93 pp | -170 | 45 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 946 | 453 | 493 | 47.89% | 50.42% | 47.50% | 2.11 pp | -40 | 49 | -0.82 |
| BTC Hourly | transformer | Transformer | 946 | 442 | 504 | 46.72% | 46.25% | 45.00% | 3.28 pp | -62 | 49 | -1.27 |
| BTC Hourly | nn | NN | 946 | 420 | 526 | 44.40% | 42.50% | 42.92% | 5.60 pp | -106 | 49 | -2.16 |
| BTC Hourly | rf | RandomForest | 946 | 419 | 527 | 44.29% | 44.17% | 43.75% | 5.71 pp | -108 | 49 | -2.20 |
| BTC Hourly | lstm | LSTM | 946 | 405 | 541 | 42.81% | 37.08% | 42.29% | 7.19 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 946 | 396 | 550 | 41.86% | 40.42% | 40.62% | 8.14 pp | -154 | 49 | -3.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 768 | 371 | 397 | 48.31% | 46.67% | 48.33% | 1.69 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 768 | 359 | 409 | 46.74% | 41.67% | 47.08% | 3.26 pp | -50 | 45 | -1.11 |
| BTC Daily | nn | NN | 768 | 356 | 412 | 46.35% | 45.00% | 46.04% | 3.65 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 768 | 325 | 443 | 42.32% | 36.25% | 40.42% | 7.68 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 768 | 322 | 446 | 41.93% | 38.75% | 42.29% | 8.07 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 778 | 304 | 474 | 39.07% | 35.00% | 36.25% | 10.93 pp | -170 | 45 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 540 | 262 | 278 | 48.52% | 46.25% | 48.12% | 1.48 pp | -16 | 51 | -0.31 |
| BTC Market Hours | transformer | Transformer | 540 | 258 | 282 | 47.78% | 48.75% | 48.12% | 2.22 pp | -24 | 51 | -0.47 |
| BTC Market Hours | nn | NN | 540 | 255 | 285 | 47.22% | 50.00% | 48.96% | 2.78 pp | -30 | 51 | -0.59 |
| BTC Market Hours | rf | RandomForest | 540 | 233 | 307 | 43.15% | 45.00% | 43.33% | 6.85 pp | -74 | 51 | -1.45 |
| BTC Market Hours | lstm | LSTM | 540 | 232 | 308 | 42.96% | 41.67% | 43.96% | 7.04 pp | -76 | 51 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 540 | 224 | 316 | 41.48% | 42.92% | 41.88% | 8.52 pp | -92 | 51 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 594 | 282 | 312 | 47.47% | 50.83% | 48.75% | 2.53 pp | -30 | 51 | -0.59 |
| BTC Market Hours Daily | nn | NN | 594 | 276 | 318 | 46.46% | 46.25% | 47.92% | 3.54 pp | -42 | 51 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 594 | 275 | 319 | 46.30% | 50.42% | 47.08% | 3.70 pp | -44 | 51 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 594 | 248 | 346 | 41.75% | 45.00% | 41.46% | 8.25 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 594 | 240 | 354 | 40.40% | 38.75% | 40.00% | 9.60 pp | -114 | 51 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 594 | 237 | 357 | 39.90% | 41.25% | 39.17% | 10.10 pp | -120 | 51 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Hourly | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 185 | 92 | 93 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 185 | 88 | 97 | 47.57% | 47.57% | 47.57% | 2.43 pp | -9 | 13 | -0.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 185 | 82 | 103 | 44.32% | 44.32% | 44.32% | 5.68 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 185 | 81 | 104 | 43.78% | 43.78% | 43.78% | 6.22 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
