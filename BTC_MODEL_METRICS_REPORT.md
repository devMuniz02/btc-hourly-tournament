# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T23:38:33.176176+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1251 | 963 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1127 | 762 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 22:00:00+00:00 | 819 | 524 | 294 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 22:00:00+00:00 | 821 | 578 | 241 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 170 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 170 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 170 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 171 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 524 | 255 | 269 | 48.66% | 45.42% | 48.75% | 1.34 pp | -14 | 50 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 170 | 83 | 87 | 48.82% | 48.82% | 48.82% | 1.18 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 170 | 83 | 87 | 48.82% | 48.82% | 48.82% | 1.18 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 524 | 251 | 273 | 47.90% | 47.50% | 48.54% | 2.10 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 752 | 366 | 386 | 48.67% | 47.92% | 48.96% | 1.33 pp | -20 | 44 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 578 | 275 | 303 | 47.58% | 50.83% | 48.75% | 2.42 pp | -28 | 50 | -0.56 |
| BTC Market Hours | nn | NN | 524 | 247 | 277 | 47.14% | 50.42% | 48.54% | 2.86 pp | -30 | 50 | -0.60 |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 578 | 270 | 308 | 46.71% | 46.25% | 47.92% | 3.29 pp | -38 | 50 | -0.76 |
| BTC Daily | transformer | Transformer | 752 | 356 | 396 | 47.34% | 44.17% | 48.96% | 2.66 pp | -40 | 44 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 929 | 442 | 487 | 47.58% | 48.75% | 46.46% | 2.42 pp | -45 | 49 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 578 | 266 | 312 | 46.02% | 50.00% | 46.67% | 3.98 pp | -46 | 50 | -0.92 |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 929 | 436 | 493 | 46.93% | 46.25% | 45.42% | 3.07 pp | -57 | 49 | -1.16 |
| BTC Daily | nn | NN | 752 | 350 | 402 | 46.54% | 45.00% | 46.88% | 3.46 pp | -52 | 44 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 170 | 77 | 93 | 45.29% | 45.29% | 45.29% | 4.71 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 170 | 77 | 93 | 45.29% | 45.29% | 45.29% | 4.71 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 524 | 228 | 296 | 43.51% | 42.50% | 44.17% | 6.49 pp | -68 | 50 | -1.36 |
| BTC Market Hours | rf | RandomForest | 524 | 225 | 299 | 42.94% | 44.58% | 43.75% | 7.06 pp | -74 | 50 | -1.48 |
| Consolidated Hourly | lstm | LSTM | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 524 | 217 | 307 | 41.41% | 42.92% | 42.08% | 8.59 pp | -90 | 50 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 578 | 240 | 338 | 41.52% | 43.33% | 41.04% | 8.48 pp | -98 | 50 | -1.96 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 929 | 414 | 515 | 44.56% | 44.17% | 44.38% | 5.44 pp | -101 | 49 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 578 | 236 | 342 | 40.83% | 41.25% | 40.62% | 9.17 pp | -106 | 50 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 170 | 72 | 98 | 42.35% | 42.35% | 42.35% | 7.65 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 170 | 72 | 98 | 42.35% | 42.35% | 42.35% | 7.65 pp | -26 | 12 | -2.17 |
| BTC Hourly | nn | NN | 929 | 411 | 518 | 44.24% | 42.08% | 41.88% | 5.76 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | xgb | XGBoost | 578 | 231 | 347 | 39.97% | 41.67% | 39.38% | 10.03 pp | -116 | 50 | -2.32 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 752 | 319 | 433 | 42.42% | 35.42% | 40.83% | 7.58 pp | -114 | 44 | -2.59 |
| BTC Daily | rf | RandomForest | 752 | 316 | 436 | 42.02% | 38.33% | 42.08% | 7.98 pp | -120 | 44 | -2.73 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 929 | 396 | 533 | 42.63% | 37.50% | 41.25% | 7.37 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 929 | 388 | 541 | 41.77% | 39.17% | 40.62% | 8.23 pp | -153 | 49 | -3.12 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |
| BTC Daily | xgb | XGBoost | 762 | 298 | 464 | 39.11% | 35.42% | 36.88% | 10.89 pp | -166 | 44 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 929 | 442 | 487 | 47.58% | 48.75% | 46.46% | 2.42 pp | -45 | 49 | -0.92 |
| BTC Hourly | transformer | Transformer | 929 | 436 | 493 | 46.93% | 46.25% | 45.42% | 3.07 pp | -57 | 49 | -1.16 |
| BTC Hourly | rf | RandomForest | 929 | 414 | 515 | 44.56% | 44.17% | 44.38% | 5.44 pp | -101 | 49 | -2.06 |
| BTC Hourly | nn | NN | 929 | 411 | 518 | 44.24% | 42.08% | 41.88% | 5.76 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 929 | 396 | 533 | 42.63% | 37.50% | 41.25% | 7.37 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 929 | 388 | 541 | 41.77% | 39.17% | 40.62% | 8.23 pp | -153 | 49 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 752 | 366 | 386 | 48.67% | 47.92% | 48.96% | 1.33 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 752 | 356 | 396 | 47.34% | 44.17% | 48.96% | 2.66 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 752 | 350 | 402 | 46.54% | 45.00% | 46.88% | 3.46 pp | -52 | 44 | -1.18 |
| BTC Daily | lstm | LSTM | 752 | 319 | 433 | 42.42% | 35.42% | 40.83% | 7.58 pp | -114 | 44 | -2.59 |
| BTC Daily | rf | RandomForest | 752 | 316 | 436 | 42.02% | 38.33% | 42.08% | 7.98 pp | -120 | 44 | -2.73 |
| BTC Daily | xgb | XGBoost | 762 | 298 | 464 | 39.11% | 35.42% | 36.88% | 10.89 pp | -166 | 44 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 524 | 255 | 269 | 48.66% | 45.42% | 48.75% | 1.34 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 524 | 251 | 273 | 47.90% | 47.50% | 48.54% | 2.10 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 524 | 247 | 277 | 47.14% | 50.42% | 48.54% | 2.86 pp | -30 | 50 | -0.60 |
| BTC Market Hours | lstm | LSTM | 524 | 228 | 296 | 43.51% | 42.50% | 44.17% | 6.49 pp | -68 | 50 | -1.36 |
| BTC Market Hours | rf | RandomForest | 524 | 225 | 299 | 42.94% | 44.58% | 43.75% | 7.06 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 524 | 217 | 307 | 41.41% | 42.92% | 42.08% | 8.59 pp | -90 | 50 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 578 | 275 | 303 | 47.58% | 50.83% | 48.75% | 2.42 pp | -28 | 50 | -0.56 |
| BTC Market Hours Daily | nn | NN | 578 | 270 | 308 | 46.71% | 46.25% | 47.92% | 3.29 pp | -38 | 50 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 578 | 266 | 312 | 46.02% | 50.00% | 46.67% | 3.98 pp | -46 | 50 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 578 | 240 | 338 | 41.52% | 43.33% | 41.04% | 8.48 pp | -98 | 50 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 578 | 236 | 342 | 40.83% | 41.25% | 40.62% | 9.17 pp | -106 | 50 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 578 | 231 | 347 | 39.97% | 41.67% | 39.38% | 10.03 pp | -116 | 50 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 170 | 83 | 87 | 48.82% | 48.82% | 48.82% | 1.18 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 12 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 170 | 77 | 93 | 45.29% | 45.29% | 45.29% | 4.71 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 170 | 72 | 98 | 42.35% | 42.35% | 42.35% | 7.65 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 170 | 83 | 87 | 48.82% | 48.82% | 48.82% | 1.18 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 170 | 82 | 88 | 48.24% | 48.24% | 48.24% | 1.76 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 170 | 77 | 93 | 45.29% | 45.29% | 45.29% | 4.71 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 170 | 72 | 98 | 42.35% | 42.35% | 42.35% | 7.65 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
