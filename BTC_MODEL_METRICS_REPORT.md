# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T19:33:04.969311+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1265 | 977 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1140 | 775 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 18:00:00+00:00 | 841 | 537 | 303 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 18:00:00+00:00 | 843 | 591 | 250 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 537 | 261 | 276 | 48.60% | 46.25% | 48.33% | 1.40 pp | -15 | 51 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 537 | 257 | 280 | 47.86% | 49.17% | 48.54% | 2.14 pp | -23 | 51 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 765 | 371 | 394 | 48.50% | 47.50% | 48.75% | 1.50 pp | -23 | 45 | -0.51 |
| BTC Market Hours | nn | NN | 537 | 255 | 282 | 47.49% | 50.83% | 48.96% | 2.51 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 591 | 281 | 310 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours Daily | nn | NN | 591 | 276 | 315 | 46.70% | 46.25% | 48.33% | 3.30 pp | -39 | 51 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 591 | 275 | 316 | 46.53% | 50.83% | 47.50% | 3.47 pp | -41 | 51 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 943 | 450 | 493 | 47.72% | 50.00% | 46.88% | 2.28 pp | -43 | 49 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| BTC Daily | transformer | Transformer | 765 | 359 | 406 | 46.93% | 42.50% | 47.50% | 3.07 pp | -47 | 45 | -1.04 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 943 | 442 | 501 | 46.87% | 46.67% | 45.00% | 3.13 pp | -59 | 49 | -1.20 |
| BTC Daily | nn | NN | 765 | 354 | 411 | 46.27% | 44.58% | 46.04% | 3.73 pp | -57 | 45 | -1.27 |
| BTC Market Hours | rf | RandomForest | 537 | 232 | 305 | 43.20% | 45.00% | 43.54% | 6.80 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 537 | 231 | 306 | 43.02% | 41.67% | 43.96% | 6.98 pp | -75 | 51 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 537 | 223 | 314 | 41.53% | 43.33% | 42.08% | 8.47 pp | -91 | 51 | -1.78 |
| BTC Market Hours Daily | rf | RandomForest | 591 | 247 | 344 | 41.79% | 45.00% | 41.67% | 8.21 pp | -97 | 51 | -1.90 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 943 | 419 | 524 | 44.43% | 42.92% | 42.71% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 943 | 419 | 524 | 44.43% | 44.58% | 43.75% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 591 | 239 | 352 | 40.44% | 39.17% | 40.00% | 9.56 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 591 | 236 | 355 | 39.93% | 41.25% | 38.96% | 10.07 pp | -119 | 51 | -2.33 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 765 | 324 | 441 | 42.35% | 35.83% | 40.62% | 7.65 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 765 | 320 | 445 | 41.83% | 37.92% | 42.29% | 8.17 pp | -125 | 45 | -2.78 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| BTC Hourly | lstm | LSTM | 943 | 402 | 541 | 42.63% | 36.25% | 41.88% | 7.37 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 943 | 396 | 547 | 41.99% | 40.83% | 40.62% | 8.01 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Daily | xgb | XGBoost | 775 | 304 | 471 | 39.23% | 35.42% | 36.67% | 10.77 pp | -167 | 45 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 943 | 450 | 493 | 47.72% | 50.00% | 46.88% | 2.28 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 943 | 442 | 501 | 46.87% | 46.67% | 45.00% | 3.13 pp | -59 | 49 | -1.20 |
| BTC Hourly | nn | NN | 943 | 419 | 524 | 44.43% | 42.92% | 42.71% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 943 | 419 | 524 | 44.43% | 44.58% | 43.75% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 943 | 402 | 541 | 42.63% | 36.25% | 41.88% | 7.37 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 943 | 396 | 547 | 41.99% | 40.83% | 40.62% | 8.01 pp | -151 | 49 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 765 | 371 | 394 | 48.50% | 47.50% | 48.75% | 1.50 pp | -23 | 45 | -0.51 |
| BTC Daily | transformer | Transformer | 765 | 359 | 406 | 46.93% | 42.50% | 47.50% | 3.07 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 765 | 354 | 411 | 46.27% | 44.58% | 46.04% | 3.73 pp | -57 | 45 | -1.27 |
| BTC Daily | lstm | LSTM | 765 | 324 | 441 | 42.35% | 35.83% | 40.62% | 7.65 pp | -117 | 45 | -2.60 |
| BTC Daily | rf | RandomForest | 765 | 320 | 445 | 41.83% | 37.92% | 42.29% | 8.17 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 775 | 304 | 471 | 39.23% | 35.42% | 36.67% | 10.77 pp | -167 | 45 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 537 | 261 | 276 | 48.60% | 46.25% | 48.33% | 1.40 pp | -15 | 51 | -0.29 |
| BTC Market Hours | transformer | Transformer | 537 | 257 | 280 | 47.86% | 49.17% | 48.54% | 2.14 pp | -23 | 51 | -0.45 |
| BTC Market Hours | nn | NN | 537 | 255 | 282 | 47.49% | 50.83% | 48.96% | 2.51 pp | -27 | 51 | -0.53 |
| BTC Market Hours | rf | RandomForest | 537 | 232 | 305 | 43.20% | 45.00% | 43.54% | 6.80 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 537 | 231 | 306 | 43.02% | 41.67% | 43.96% | 6.98 pp | -75 | 51 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 537 | 223 | 314 | 41.53% | 43.33% | 42.08% | 8.47 pp | -91 | 51 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 591 | 281 | 310 | 47.55% | 51.25% | 48.75% | 2.45 pp | -29 | 51 | -0.57 |
| BTC Market Hours Daily | nn | NN | 591 | 276 | 315 | 46.70% | 46.25% | 48.33% | 3.30 pp | -39 | 51 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 591 | 275 | 316 | 46.53% | 50.83% | 47.50% | 3.47 pp | -41 | 51 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 591 | 247 | 344 | 41.79% | 45.00% | 41.67% | 8.21 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 591 | 239 | 352 | 40.44% | 39.17% | 40.00% | 9.56 pp | -113 | 51 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 591 | 236 | 355 | 39.93% | 41.25% | 38.96% | 10.07 pp | -119 | 51 | -2.33 |

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
