# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T18:33:19.414423+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1215 | 927 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1091 | 726 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 17:00:00+00:00 | 752 | 488 | 263 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 17:00:00+00:00 | 754 | 542 | 210 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 488 | 235 | 253 | 48.16% | 43.75% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 716 | 347 | 369 | 48.46% | 47.08% | 48.54% | 1.54 pp | -22 | 43 | -0.51 |
| BTC Market Hours | nn | NN | 488 | 229 | 259 | 46.93% | 48.75% | 47.29% | 3.07 pp | -30 | 47 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 488 | 228 | 260 | 46.72% | 42.50% | 47.08% | 3.28 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 716 | 341 | 375 | 47.63% | 45.42% | 49.79% | 2.37 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 542 | 251 | 291 | 46.31% | 49.58% | 47.29% | 3.69 pp | -40 | 47 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 893 | 425 | 468 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 542 | 248 | 294 | 45.76% | 47.92% | 46.88% | 4.24 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 542 | 248 | 294 | 45.76% | 44.58% | 46.88% | 4.24 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 893 | 423 | 470 | 47.37% | 48.33% | 47.29% | 2.63 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 716 | 331 | 385 | 46.23% | 43.75% | 47.71% | 3.77 pp | -54 | 43 | -1.26 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 488 | 210 | 278 | 43.03% | 40.83% | 42.92% | 6.97 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 488 | 209 | 279 | 42.83% | 42.08% | 42.92% | 7.17 pp | -70 | 47 | -1.49 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 488 | 198 | 290 | 40.57% | 39.58% | 40.42% | 9.43 pp | -92 | 47 | -1.96 |
| BTC Hourly | nn | NN | 893 | 399 | 494 | 44.68% | 45.42% | 42.50% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | rf | RandomForest | 893 | 399 | 494 | 44.68% | 45.83% | 44.17% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 542 | 223 | 319 | 41.14% | 41.67% | 41.25% | 8.86 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 542 | 218 | 324 | 40.22% | 38.33% | 40.62% | 9.78 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 716 | 309 | 407 | 43.16% | 37.50% | 42.29% | 6.84 pp | -98 | 43 | -2.28 |
| Consolidated Market Hours Daily | nn | NN | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 542 | 215 | 327 | 39.67% | 39.58% | 39.38% | 10.33 pp | -112 | 47 | -2.38 |
| BTC Daily | rf | RandomForest | 716 | 306 | 410 | 42.74% | 40.83% | 43.54% | 7.26 pp | -104 | 43 | -2.42 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 893 | 383 | 510 | 42.89% | 39.58% | 42.29% | 7.11 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 893 | 377 | 516 | 42.22% | 42.92% | 42.08% | 7.78 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 726 | 287 | 439 | 39.53% | 35.00% | 38.75% | 10.47 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 893 | 425 | 468 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 893 | 423 | 470 | 47.37% | 48.33% | 47.29% | 2.63 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 893 | 399 | 494 | 44.68% | 45.42% | 42.50% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | rf | RandomForest | 893 | 399 | 494 | 44.68% | 45.83% | 44.17% | 5.32 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 893 | 383 | 510 | 42.89% | 39.58% | 42.29% | 7.11 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 893 | 377 | 516 | 42.22% | 42.92% | 42.08% | 7.78 pp | -139 | 47 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 716 | 347 | 369 | 48.46% | 47.08% | 48.54% | 1.54 pp | -22 | 43 | -0.51 |
| BTC Daily | transformer | Transformer | 716 | 341 | 375 | 47.63% | 45.42% | 49.79% | 2.37 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 716 | 331 | 385 | 46.23% | 43.75% | 47.71% | 3.77 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 716 | 309 | 407 | 43.16% | 37.50% | 42.29% | 6.84 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 716 | 306 | 410 | 42.74% | 40.83% | 43.54% | 7.26 pp | -104 | 43 | -2.42 |
| BTC Daily | xgb | XGBoost | 726 | 287 | 439 | 39.53% | 35.00% | 38.75% | 10.47 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 488 | 235 | 253 | 48.16% | 43.75% | 48.12% | 1.84 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 488 | 229 | 259 | 46.93% | 48.75% | 47.29% | 3.07 pp | -30 | 47 | -0.64 |
| BTC Market Hours | transformer | Transformer | 488 | 228 | 260 | 46.72% | 42.50% | 47.08% | 3.28 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 488 | 210 | 278 | 43.03% | 40.83% | 42.92% | 6.97 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 488 | 209 | 279 | 42.83% | 42.08% | 42.92% | 7.17 pp | -70 | 47 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 488 | 198 | 290 | 40.57% | 39.58% | 40.42% | 9.43 pp | -92 | 47 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 542 | 251 | 291 | 46.31% | 49.58% | 47.29% | 3.69 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 542 | 248 | 294 | 45.76% | 47.92% | 46.88% | 4.24 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 542 | 248 | 294 | 45.76% | 44.58% | 46.88% | 4.24 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 542 | 223 | 319 | 41.14% | 41.67% | 41.25% | 8.86 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 542 | 218 | 324 | 40.22% | 38.33% | 40.62% | 9.78 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 542 | 215 | 327 | 39.67% | 39.58% | 39.38% | 10.33 pp | -112 | 47 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
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
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
