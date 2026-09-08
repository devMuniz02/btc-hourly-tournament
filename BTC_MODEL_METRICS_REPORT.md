# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T13:37:56.633060+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1293 | 1005 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1169 | 804 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 12:00:00+00:00 | 890 | 566 | 323 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 12:00:00+00:00 | 892 | 620 | 270 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 208 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 208 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 208 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 209 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 566 | 275 | 291 | 48.59% | 47.08% | 47.71% | 1.41 pp | -16 | 53 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 208 | 101 | 107 | 48.56% | 48.56% | 48.56% | 1.44 pp | -6 | 14 | -0.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 208 | 101 | 107 | 48.56% | 48.56% | 48.56% | 1.44 pp | -6 | 14 | -0.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 384 | 410 | 48.36% | 46.67% | 47.92% | 1.64 pp | -26 | 46 | -0.57 |
| BTC Market Hours | nn | NN | 566 | 268 | 298 | 47.35% | 50.83% | 48.75% | 2.65 pp | -30 | 53 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 14 | -0.57 |
| BTC Market Hours | transformer | Transformer | 566 | 266 | 300 | 47.00% | 46.67% | 46.88% | 3.00 pp | -34 | 53 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 620 | 290 | 330 | 46.77% | 49.17% | 47.71% | 3.23 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | nn | NN | 620 | 288 | 332 | 46.45% | 47.08% | 47.50% | 3.55 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 620 | 287 | 333 | 46.29% | 48.33% | 46.88% | 3.71 pp | -46 | 53 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 971 | 460 | 511 | 47.37% | 48.75% | 46.04% | 2.63 pp | -51 | 50 | -1.02 |
| Consolidated Hourly | xgb | XGBoost | 208 | 96 | 112 | 46.15% | 46.15% | 46.15% | 3.85 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 208 | 96 | 112 | 46.15% | 46.15% | 46.15% | 3.85 pp | -16 | 14 | -1.14 |
| BTC Daily | nn | NN | 794 | 369 | 425 | 46.47% | 45.00% | 45.00% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.00% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| Consolidated Hourly | lstm | LSTM | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 971 | 452 | 519 | 46.55% | 44.17% | 43.75% | 3.45 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 566 | 245 | 321 | 43.29% | 42.08% | 43.75% | 6.71 pp | -76 | 53 | -1.43 |
| BTC Market Hours | rf | RandomForest | 566 | 244 | 322 | 43.11% | 45.42% | 43.54% | 6.89 pp | -78 | 53 | -1.47 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 566 | 243 | 323 | 42.93% | 46.67% | 43.54% | 7.07 pp | -80 | 53 | -1.51 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 14 | -1.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 620 | 258 | 362 | 41.61% | 43.75% | 40.42% | 8.39 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 620 | 255 | 365 | 41.13% | 44.17% | 40.62% | 8.87 pp | -110 | 53 | -2.08 |
| Consolidated Hourly | transformer | Transformer | 208 | 89 | 119 | 42.79% | 42.79% | 42.79% | 7.21 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 208 | 89 | 119 | 42.79% | 42.79% | 42.79% | 7.21 pp | -30 | 14 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 620 | 252 | 368 | 40.65% | 40.42% | 39.79% | 9.35 pp | -116 | 53 | -2.19 |
| BTC Hourly | nn | NN | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 794 | 333 | 461 | 41.94% | 33.75% | 39.79% | 8.06 pp | -128 | 46 | -2.78 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.50% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Hourly | lstm | LSTM | 971 | 413 | 558 | 42.53% | 37.50% | 41.04% | 7.47 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 971 | 400 | 571 | 41.19% | 35.83% | 38.54% | 8.81 pp | -171 | 50 | -3.42 |
| BTC Daily | xgb | XGBoost | 804 | 314 | 490 | 39.05% | 35.42% | 36.04% | 10.95 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 971 | 460 | 511 | 47.37% | 48.75% | 46.04% | 2.63 pp | -51 | 50 | -1.02 |
| BTC Hourly | transformer | Transformer | 971 | 452 | 519 | 46.55% | 44.17% | 43.75% | 3.45 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 971 | 429 | 542 | 44.18% | 41.25% | 42.50% | 5.82 pp | -113 | 50 | -2.26 |
| BTC Hourly | lstm | LSTM | 971 | 413 | 558 | 42.53% | 37.50% | 41.04% | 7.47 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 971 | 400 | 571 | 41.19% | 35.83% | 38.54% | 8.81 pp | -171 | 50 | -3.42 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 794 | 384 | 410 | 48.36% | 46.67% | 47.92% | 1.64 pp | -26 | 46 | -0.57 |
| BTC Daily | nn | NN | 794 | 369 | 425 | 46.47% | 45.00% | 45.00% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 794 | 369 | 425 | 46.47% | 40.00% | 46.46% | 3.53 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 794 | 333 | 461 | 41.94% | 33.75% | 39.79% | 8.06 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 794 | 331 | 463 | 41.69% | 37.50% | 41.46% | 8.31 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 804 | 314 | 490 | 39.05% | 35.42% | 36.04% | 10.95 pp | -176 | 46 | -3.83 |

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
| Consolidated Hourly | rf | RandomForest | 208 | 101 | 107 | 48.56% | 48.56% | 48.56% | 1.44 pp | -6 | 14 | -0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 208 | 96 | 112 | 46.15% | 46.15% | 46.15% | 3.85 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | nn | NN | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 208 | 89 | 119 | 42.79% | 42.79% | 42.79% | 7.21 pp | -30 | 14 | -2.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 208 | 101 | 107 | 48.56% | 48.56% | 48.56% | 1.44 pp | -6 | 14 | -0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 208 | 96 | 112 | 46.15% | 46.15% | 46.15% | 3.85 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 208 | 89 | 119 | 42.79% | 42.79% | 42.79% | 7.21 pp | -30 | 14 | -2.14 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
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
