# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T14:28:59.976377+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1294 | 1006 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1169 | 804 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 891 | 566 | 324 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 893 | 620 | 271 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 566 | 275 | 291 | 48.59% | 47.08% | 47.71% | 1.41 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 566 | 268 | 298 | 47.35% | 50.83% | 48.75% | 2.65 pp | -30 | 53 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 383 | 411 | 48.24% | 46.67% | 47.71% | 1.76 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 566 | 266 | 300 | 47.00% | 46.67% | 46.88% | 3.00 pp | -34 | 53 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 620 | 290 | 330 | 46.77% | 49.17% | 47.71% | 3.23 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 620 | 288 | 332 | 46.45% | 47.08% | 47.50% | 3.55 pp | -44 | 53 | -0.83 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 620 | 287 | 333 | 46.29% | 48.33% | 46.88% | 3.71 pp | -46 | 53 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 972 | 461 | 511 | 47.43% | 48.75% | 46.25% | 2.57 pp | -50 | 50 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.42% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 794 | 368 | 426 | 46.35% | 45.00% | 44.79% | 3.65 pp | -58 | 46 | -1.26 |
| BTC Hourly | transformer | Transformer | 972 | 453 | 519 | 46.60% | 44.58% | 43.96% | 3.40 pp | -66 | 50 | -1.32 |
| BTC Market Hours | lstm | LSTM | 566 | 245 | 321 | 43.29% | 42.08% | 43.75% | 6.71 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 566 | 244 | 322 | 43.11% | 45.42% | 43.54% | 6.89 pp | -78 | 53 | -1.47 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 566 | 243 | 323 | 42.93% | 46.67% | 43.54% | 7.07 pp | -80 | 53 | -1.51 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 620 | 258 | 362 | 41.61% | 43.75% | 40.42% | 8.39 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 620 | 255 | 365 | 41.13% | 44.17% | 40.62% | 8.87 pp | -110 | 53 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 620 | 252 | 368 | 40.65% | 40.42% | 39.79% | 9.35 pp | -116 | 53 | -2.19 |
| BTC Hourly | rf | RandomForest | 972 | 430 | 542 | 44.24% | 41.67% | 42.71% | 5.76 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 972 | 429 | 543 | 44.14% | 40.83% | 42.50% | 5.86 pp | -114 | 50 | -2.28 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 794 | 334 | 460 | 42.07% | 34.17% | 40.00% | 7.93 pp | -126 | 46 | -2.74 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.92% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Hourly | lstm | LSTM | 972 | 413 | 559 | 42.49% | 37.50% | 41.04% | 7.51 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 972 | 401 | 571 | 41.26% | 35.83% | 38.75% | 8.74 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 804 | 313 | 491 | 38.93% | 35.42% | 35.83% | 11.07 pp | -178 | 46 | -3.87 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 972 | 461 | 511 | 47.43% | 48.75% | 46.25% | 2.57 pp | -50 | 50 | -1.00 |
| BTC Hourly | transformer | Transformer | 972 | 453 | 519 | 46.60% | 44.58% | 43.96% | 3.40 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 972 | 430 | 542 | 44.24% | 41.67% | 42.71% | 5.76 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 972 | 429 | 543 | 44.14% | 40.83% | 42.50% | 5.86 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 972 | 413 | 559 | 42.49% | 37.50% | 41.04% | 7.51 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 972 | 401 | 571 | 41.26% | 35.83% | 38.75% | 8.74 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 383 | 411 | 48.24% | 46.67% | 47.71% | 1.76 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.42% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | nn | NN | 794 | 368 | 426 | 46.35% | 45.00% | 44.79% | 3.65 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 794 | 334 | 460 | 42.07% | 34.17% | 40.00% | 7.93 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.92% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 804 | 313 | 491 | 38.93% | 35.42% | 35.83% | 11.07 pp | -178 | 46 | -3.87 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 566 | 275 | 291 | 48.59% | 47.08% | 47.71% | 1.41 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 566 | 268 | 298 | 47.35% | 50.83% | 48.75% | 2.65 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 566 | 266 | 300 | 47.00% | 46.67% | 46.88% | 3.00 pp | -34 | 53 | -0.64 |
| BTC Market Hours | lstm | LSTM | 566 | 245 | 321 | 43.29% | 42.08% | 43.75% | 6.71 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 566 | 244 | 322 | 43.11% | 45.42% | 43.54% | 6.89 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 566 | 243 | 323 | 42.93% | 46.67% | 43.54% | 7.07 pp | -80 | 53 | -1.51 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 620 | 290 | 330 | 46.77% | 49.17% | 47.71% | 3.23 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 620 | 288 | 332 | 46.45% | 47.08% | 47.50% | 3.55 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 620 | 287 | 333 | 46.29% | 48.33% | 46.88% | 3.71 pp | -46 | 53 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 620 | 258 | 362 | 41.61% | 43.75% | 40.42% | 8.39 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 620 | 255 | 365 | 41.13% | 44.17% | 40.62% | 8.87 pp | -110 | 53 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 620 | 252 | 368 | 40.65% | 40.42% | 39.79% | 9.35 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
