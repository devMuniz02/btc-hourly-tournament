# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T17:24:08.409068+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1263 | 975 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1139 | 774 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 16:00:00+00:00 | 838 | 536 | 301 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 16:00:00+00:00 | 840 | 590 | 248 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 182 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 536 | 261 | 275 | 48.69% | 46.25% | 48.54% | 1.31 pp | -14 | 51 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 536 | 257 | 279 | 47.95% | 49.17% | 48.75% | 2.05 pp | -22 | 51 | -0.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 764 | 372 | 392 | 48.69% | 47.92% | 48.96% | 1.31 pp | -20 | 45 | -0.44 |
| BTC Market Hours | nn | NN | 536 | 255 | 281 | 47.57% | 50.83% | 49.17% | 2.43 pp | -26 | 51 | -0.51 |
| Consolidated Hourly | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 590 | 281 | 309 | 47.63% | 51.67% | 48.75% | 2.37 pp | -28 | 51 | -0.55 |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 590 | 276 | 314 | 46.78% | 46.25% | 48.33% | 3.22 pp | -38 | 51 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 590 | 275 | 315 | 46.61% | 50.83% | 47.50% | 3.39 pp | -40 | 51 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 941 | 448 | 493 | 47.61% | 50.00% | 46.46% | 2.39 pp | -45 | 49 | -0.92 |
| BTC Daily | transformer | Transformer | 764 | 360 | 404 | 47.12% | 42.92% | 47.71% | 2.88 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| BTC Daily | nn | NN | 764 | 355 | 409 | 46.47% | 45.00% | 46.46% | 3.53 pp | -54 | 45 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 941 | 441 | 500 | 46.87% | 46.67% | 45.21% | 3.13 pp | -59 | 49 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 536 | 232 | 304 | 43.28% | 45.00% | 43.75% | 6.72 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 536 | 231 | 305 | 43.10% | 41.67% | 43.96% | 6.90 pp | -74 | 51 | -1.45 |
| Consolidated Hourly | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 536 | 223 | 313 | 41.60% | 43.33% | 42.29% | 8.40 pp | -90 | 51 | -1.76 |
| BTC Market Hours Daily | rf | RandomForest | 590 | 246 | 344 | 41.69% | 45.00% | 41.46% | 8.31 pp | -98 | 51 | -1.92 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 941 | 418 | 523 | 44.42% | 42.92% | 42.50% | 5.58 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 941 | 418 | 523 | 44.42% | 44.58% | 43.54% | 5.58 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 590 | 239 | 351 | 40.51% | 39.17% | 40.21% | 9.49 pp | -112 | 51 | -2.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 590 | 236 | 354 | 40.00% | 41.67% | 38.96% | 10.00 pp | -118 | 51 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 764 | 323 | 441 | 42.28% | 35.42% | 40.62% | 7.72 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 764 | 320 | 444 | 41.88% | 37.92% | 42.29% | 8.12 pp | -124 | 45 | -2.76 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| BTC Hourly | lstm | LSTM | 941 | 401 | 540 | 42.61% | 36.25% | 41.88% | 7.39 pp | -139 | 49 | -2.84 |
| Consolidated Market Hours Daily | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 941 | 395 | 546 | 41.98% | 40.83% | 40.42% | 8.02 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |
| BTC Daily | xgb | XGBoost | 774 | 304 | 470 | 39.28% | 35.42% | 36.88% | 10.72 pp | -166 | 45 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 941 | 448 | 493 | 47.61% | 50.00% | 46.46% | 2.39 pp | -45 | 49 | -0.92 |
| BTC Hourly | transformer | Transformer | 941 | 441 | 500 | 46.87% | 46.67% | 45.21% | 3.13 pp | -59 | 49 | -1.20 |
| BTC Hourly | nn | NN | 941 | 418 | 523 | 44.42% | 42.92% | 42.50% | 5.58 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 941 | 418 | 523 | 44.42% | 44.58% | 43.54% | 5.58 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 941 | 401 | 540 | 42.61% | 36.25% | 41.88% | 7.39 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 941 | 395 | 546 | 41.98% | 40.83% | 40.42% | 8.02 pp | -151 | 49 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 764 | 372 | 392 | 48.69% | 47.92% | 48.96% | 1.31 pp | -20 | 45 | -0.44 |
| BTC Daily | transformer | Transformer | 764 | 360 | 404 | 47.12% | 42.92% | 47.71% | 2.88 pp | -44 | 45 | -0.98 |
| BTC Daily | nn | NN | 764 | 355 | 409 | 46.47% | 45.00% | 46.46% | 3.53 pp | -54 | 45 | -1.20 |
| BTC Daily | lstm | LSTM | 764 | 323 | 441 | 42.28% | 35.42% | 40.62% | 7.72 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 764 | 320 | 444 | 41.88% | 37.92% | 42.29% | 8.12 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 774 | 304 | 470 | 39.28% | 35.42% | 36.88% | 10.72 pp | -166 | 45 | -3.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 536 | 261 | 275 | 48.69% | 46.25% | 48.54% | 1.31 pp | -14 | 51 | -0.27 |
| BTC Market Hours | transformer | Transformer | 536 | 257 | 279 | 47.95% | 49.17% | 48.75% | 2.05 pp | -22 | 51 | -0.43 |
| BTC Market Hours | nn | NN | 536 | 255 | 281 | 47.57% | 50.83% | 49.17% | 2.43 pp | -26 | 51 | -0.51 |
| BTC Market Hours | rf | RandomForest | 536 | 232 | 304 | 43.28% | 45.00% | 43.75% | 6.72 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 536 | 231 | 305 | 43.10% | 41.67% | 43.96% | 6.90 pp | -74 | 51 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 536 | 223 | 313 | 41.60% | 43.33% | 42.29% | 8.40 pp | -90 | 51 | -1.76 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 590 | 281 | 309 | 47.63% | 51.67% | 48.75% | 2.37 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | nn | NN | 590 | 276 | 314 | 46.78% | 46.25% | 48.33% | 3.22 pp | -38 | 51 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 590 | 275 | 315 | 46.61% | 50.83% | 47.50% | 3.39 pp | -40 | 51 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 590 | 246 | 344 | 41.69% | 45.00% | 41.46% | 8.31 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 590 | 239 | 351 | 40.51% | 39.17% | 40.21% | 9.49 pp | -112 | 51 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 590 | 236 | 354 | 40.00% | 41.67% | 38.96% | 10.00 pp | -118 | 51 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
