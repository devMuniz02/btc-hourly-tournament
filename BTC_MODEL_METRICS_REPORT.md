# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T16:02:00.675188+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1262 | 974 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1138 | 773 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 15:00:00+00:00 | 836 | 535 | 300 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 15:00:00+00:00 | 838 | 589 | 247 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 180 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 535 | 260 | 275 | 48.60% | 45.83% | 48.54% | 1.40 pp | -15 | 51 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 535 | 257 | 278 | 48.04% | 49.17% | 48.75% | 1.96 pp | -21 | 51 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 763 | 371 | 392 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 45 | -0.47 |
| BTC Market Hours | nn | NN | 535 | 254 | 281 | 47.48% | 50.42% | 49.17% | 2.52 pp | -27 | 51 | -0.53 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 589 | 280 | 309 | 47.54% | 51.25% | 48.75% | 2.46 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 589 | 275 | 314 | 46.69% | 46.25% | 48.12% | 3.31 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 589 | 274 | 315 | 46.52% | 50.83% | 47.29% | 3.48 pp | -41 | 50 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 940 | 448 | 492 | 47.66% | 50.00% | 46.67% | 2.34 pp | -44 | 49 | -0.90 |
| BTC Daily | transformer | Transformer | 763 | 359 | 404 | 47.05% | 42.92% | 47.71% | 2.95 pp | -45 | 45 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| BTC Daily | nn | NN | 763 | 355 | 408 | 46.53% | 45.42% | 46.67% | 3.47 pp | -53 | 45 | -1.18 |
| BTC Hourly | transformer | Transformer | 940 | 441 | 499 | 46.91% | 46.67% | 45.42% | 3.09 pp | -58 | 49 | -1.18 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| BTC Market Hours | lstm | LSTM | 535 | 231 | 304 | 43.18% | 42.08% | 43.96% | 6.82 pp | -73 | 51 | -1.43 |
| BTC Market Hours | rf | RandomForest | 535 | 231 | 304 | 43.18% | 44.58% | 43.75% | 6.82 pp | -73 | 51 | -1.43 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 535 | 223 | 312 | 41.68% | 43.33% | 42.50% | 8.32 pp | -89 | 51 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 589 | 245 | 344 | 41.60% | 44.58% | 41.25% | 8.40 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 940 | 418 | 522 | 44.47% | 44.58% | 43.75% | 5.53 pp | -104 | 49 | -2.12 |
| BTC Hourly | nn | NN | 940 | 417 | 523 | 44.36% | 42.50% | 42.29% | 5.64 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 589 | 239 | 350 | 40.58% | 39.58% | 40.42% | 9.42 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 589 | 235 | 354 | 39.90% | 41.67% | 38.75% | 10.10 pp | -119 | 50 | -2.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 763 | 323 | 440 | 42.33% | 35.83% | 40.83% | 7.67 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 763 | 320 | 443 | 41.94% | 38.33% | 42.50% | 8.06 pp | -123 | 45 | -2.73 |
| BTC Hourly | lstm | LSTM | 940 | 401 | 539 | 42.66% | 36.25% | 41.88% | 7.34 pp | -138 | 49 | -2.82 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 940 | 395 | 545 | 42.02% | 40.83% | 40.62% | 7.98 pp | -150 | 49 | -3.06 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Daily | xgb | XGBoost | 773 | 304 | 469 | 39.33% | 35.42% | 37.08% | 10.67 pp | -165 | 45 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 940 | 448 | 492 | 47.66% | 50.00% | 46.67% | 2.34 pp | -44 | 49 | -0.90 |
| BTC Hourly | transformer | Transformer | 940 | 441 | 499 | 46.91% | 46.67% | 45.42% | 3.09 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 940 | 418 | 522 | 44.47% | 44.58% | 43.75% | 5.53 pp | -104 | 49 | -2.12 |
| BTC Hourly | nn | NN | 940 | 417 | 523 | 44.36% | 42.50% | 42.29% | 5.64 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 940 | 401 | 539 | 42.66% | 36.25% | 41.88% | 7.34 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 940 | 395 | 545 | 42.02% | 40.83% | 40.62% | 7.98 pp | -150 | 49 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 763 | 371 | 392 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 45 | -0.47 |
| BTC Daily | transformer | Transformer | 763 | 359 | 404 | 47.05% | 42.92% | 47.71% | 2.95 pp | -45 | 45 | -1.00 |
| BTC Daily | nn | NN | 763 | 355 | 408 | 46.53% | 45.42% | 46.67% | 3.47 pp | -53 | 45 | -1.18 |
| BTC Daily | lstm | LSTM | 763 | 323 | 440 | 42.33% | 35.83% | 40.83% | 7.67 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 763 | 320 | 443 | 41.94% | 38.33% | 42.50% | 8.06 pp | -123 | 45 | -2.73 |
| BTC Daily | xgb | XGBoost | 773 | 304 | 469 | 39.33% | 35.42% | 37.08% | 10.67 pp | -165 | 45 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 535 | 260 | 275 | 48.60% | 45.83% | 48.54% | 1.40 pp | -15 | 51 | -0.29 |
| BTC Market Hours | transformer | Transformer | 535 | 257 | 278 | 48.04% | 49.17% | 48.75% | 1.96 pp | -21 | 51 | -0.41 |
| BTC Market Hours | nn | NN | 535 | 254 | 281 | 47.48% | 50.42% | 49.17% | 2.52 pp | -27 | 51 | -0.53 |
| BTC Market Hours | lstm | LSTM | 535 | 231 | 304 | 43.18% | 42.08% | 43.96% | 6.82 pp | -73 | 51 | -1.43 |
| BTC Market Hours | rf | RandomForest | 535 | 231 | 304 | 43.18% | 44.58% | 43.75% | 6.82 pp | -73 | 51 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 535 | 223 | 312 | 41.68% | 43.33% | 42.50% | 8.32 pp | -89 | 51 | -1.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 589 | 280 | 309 | 47.54% | 51.25% | 48.75% | 2.46 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 589 | 275 | 314 | 46.69% | 46.25% | 48.12% | 3.31 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 589 | 274 | 315 | 46.52% | 50.83% | 47.29% | 3.48 pp | -41 | 50 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 589 | 245 | 344 | 41.60% | 44.58% | 41.25% | 8.40 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 589 | 239 | 350 | 40.58% | 39.58% | 40.42% | 9.42 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 589 | 235 | 354 | 39.90% | 41.67% | 38.75% | 10.10 pp | -119 | 50 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
