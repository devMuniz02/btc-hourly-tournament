# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T18:01:51.267258+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1264 | 976 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1139 | 774 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 839 | 536 | 302 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 841 | 590 | 249 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 536 | 261 | 275 | 48.69% | 46.25% | 48.54% | 1.31 pp | -14 | 51 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 536 | 257 | 279 | 47.95% | 49.17% | 48.75% | 2.05 pp | -22 | 51 | -0.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 764 | 371 | 393 | 48.56% | 47.50% | 48.75% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Market Hours | nn | NN | 536 | 255 | 281 | 47.57% | 50.83% | 49.17% | 2.43 pp | -26 | 51 | -0.51 |
| BTC Market Hours Daily | transformer | Transformer | 590 | 281 | 309 | 47.63% | 51.67% | 48.75% | 2.37 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | nn | NN | 590 | 276 | 314 | 46.78% | 46.25% | 48.33% | 3.22 pp | -38 | 51 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 590 | 275 | 315 | 46.61% | 50.83% | 47.50% | 3.39 pp | -40 | 51 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 942 | 449 | 493 | 47.66% | 50.00% | 46.67% | 2.34 pp | -44 | 49 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| BTC Daily | transformer | Transformer | 764 | 359 | 405 | 46.99% | 42.50% | 47.50% | 3.01 pp | -46 | 45 | -1.02 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 942 | 441 | 501 | 46.82% | 46.67% | 45.00% | 3.18 pp | -60 | 49 | -1.22 |
| BTC Daily | nn | NN | 764 | 354 | 410 | 46.34% | 44.58% | 46.25% | 3.66 pp | -56 | 45 | -1.24 |
| BTC Market Hours | rf | RandomForest | 536 | 232 | 304 | 43.28% | 45.00% | 43.75% | 6.72 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 536 | 231 | 305 | 43.10% | 41.67% | 43.96% | 6.90 pp | -74 | 51 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 536 | 223 | 313 | 41.60% | 43.33% | 42.29% | 8.40 pp | -90 | 51 | -1.76 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 590 | 246 | 344 | 41.69% | 45.00% | 41.46% | 8.31 pp | -98 | 51 | -1.92 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 942 | 418 | 524 | 44.37% | 42.92% | 42.50% | 5.63 pp | -106 | 49 | -2.16 |
| BTC Hourly | rf | RandomForest | 942 | 418 | 524 | 44.37% | 44.58% | 43.54% | 5.63 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 590 | 239 | 351 | 40.51% | 39.17% | 40.21% | 9.49 pp | -112 | 51 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 590 | 236 | 354 | 40.00% | 41.67% | 38.96% | 10.00 pp | -118 | 51 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 764 | 323 | 441 | 42.28% | 35.42% | 40.62% | 7.72 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 764 | 319 | 445 | 41.75% | 37.50% | 42.08% | 8.25 pp | -126 | 45 | -2.80 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| BTC Hourly | lstm | LSTM | 942 | 401 | 541 | 42.57% | 36.25% | 41.67% | 7.43 pp | -140 | 49 | -2.86 |
| BTC Hourly | xgb | XGBoost | 942 | 395 | 547 | 41.93% | 40.42% | 40.42% | 8.07 pp | -152 | 49 | -3.10 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Daily | xgb | XGBoost | 774 | 303 | 471 | 39.15% | 35.00% | 36.67% | 10.85 pp | -168 | 45 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 942 | 449 | 493 | 47.66% | 50.00% | 46.67% | 2.34 pp | -44 | 49 | -0.90 |
| BTC Hourly | transformer | Transformer | 942 | 441 | 501 | 46.82% | 46.67% | 45.00% | 3.18 pp | -60 | 49 | -1.22 |
| BTC Hourly | nn | NN | 942 | 418 | 524 | 44.37% | 42.92% | 42.50% | 5.63 pp | -106 | 49 | -2.16 |
| BTC Hourly | rf | RandomForest | 942 | 418 | 524 | 44.37% | 44.58% | 43.54% | 5.63 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 942 | 401 | 541 | 42.57% | 36.25% | 41.67% | 7.43 pp | -140 | 49 | -2.86 |
| BTC Hourly | xgb | XGBoost | 942 | 395 | 547 | 41.93% | 40.42% | 40.42% | 8.07 pp | -152 | 49 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 764 | 371 | 393 | 48.56% | 47.50% | 48.75% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Daily | transformer | Transformer | 764 | 359 | 405 | 46.99% | 42.50% | 47.50% | 3.01 pp | -46 | 45 | -1.02 |
| BTC Daily | nn | NN | 764 | 354 | 410 | 46.34% | 44.58% | 46.25% | 3.66 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 764 | 323 | 441 | 42.28% | 35.42% | 40.62% | 7.72 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 764 | 319 | 445 | 41.75% | 37.50% | 42.08% | 8.25 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 774 | 303 | 471 | 39.15% | 35.00% | 36.67% | 10.85 pp | -168 | 45 | -3.73 |

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
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
