# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T18:23:29.249468+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1343 | 1055 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1219 | 854 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 17:00:00+00:00 | 984 | 616 | 367 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 17:00:00+00:00 | 986 | 670 | 314 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 616 | 297 | 319 | 48.21% | 47.08% | 47.29% | 1.79 pp | -22 | 57 | -0.39 |
| BTC Market Hours | nn | NN | 616 | 295 | 321 | 47.89% | 50.83% | 49.79% | 2.11 pp | -26 | 57 | -0.46 |
| BTC Market Hours Daily | nn | NN | 670 | 316 | 354 | 47.16% | 50.00% | 48.54% | 2.84 pp | -38 | 57 | -0.67 |
| BTC Market Hours | transformer | Transformer | 616 | 287 | 329 | 46.59% | 46.25% | 45.62% | 3.41 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 670 | 313 | 357 | 46.72% | 48.75% | 47.08% | 3.28 pp | -44 | 57 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 670 | 311 | 359 | 46.42% | 47.92% | 47.92% | 3.58 pp | -48 | 57 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 844 | 401 | 443 | 47.51% | 43.75% | 46.04% | 2.49 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1021 | 483 | 538 | 47.31% | 46.67% | 46.25% | 2.69 pp | -55 | 53 | -1.04 |
| BTC Daily | nn | NN | 844 | 394 | 450 | 46.68% | 45.83% | 45.42% | 3.32 pp | -56 | 48 | -1.17 |
| BTC Hourly | transformer | Transformer | 1021 | 479 | 542 | 46.91% | 46.67% | 45.00% | 3.09 pp | -63 | 53 | -1.19 |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 844 | 389 | 455 | 46.09% | 37.92% | 44.38% | 3.91 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 616 | 264 | 352 | 42.86% | 42.92% | 43.12% | 7.14 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 616 | 264 | 352 | 42.86% | 43.75% | 41.88% | 7.14 pp | -88 | 57 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 616 | 262 | 354 | 42.53% | 46.67% | 42.92% | 7.47 pp | -92 | 57 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 670 | 278 | 392 | 41.49% | 43.33% | 41.67% | 8.51 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 670 | 275 | 395 | 41.04% | 44.17% | 41.25% | 8.96 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 670 | 275 | 395 | 41.04% | 43.75% | 41.25% | 8.96 pp | -120 | 57 | -2.11 |
| BTC Hourly | nn | NN | 1021 | 449 | 572 | 43.98% | 40.83% | 40.83% | 6.02 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1021 | 447 | 574 | 43.78% | 40.42% | 42.29% | 6.22 pp | -127 | 53 | -2.40 |
| BTC Daily | lstm | LSTM | 844 | 356 | 488 | 42.18% | 35.83% | 39.79% | 7.82 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| BTC Daily | rf | RandomForest | 844 | 351 | 493 | 41.59% | 37.50% | 41.04% | 8.41 pp | -142 | 48 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| BTC Hourly | lstm | LSTM | 1021 | 429 | 592 | 42.02% | 34.58% | 39.38% | 7.98 pp | -163 | 53 | -3.08 |
| BTC Hourly | xgb | XGBoost | 1021 | 419 | 602 | 41.04% | 34.58% | 37.71% | 8.96 pp | -183 | 53 | -3.45 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| BTC Daily | xgb | XGBoost | 854 | 337 | 517 | 39.46% | 37.92% | 36.25% | 10.54 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1021 | 483 | 538 | 47.31% | 46.67% | 46.25% | 2.69 pp | -55 | 53 | -1.04 |
| BTC Hourly | transformer | Transformer | 1021 | 479 | 542 | 46.91% | 46.67% | 45.00% | 3.09 pp | -63 | 53 | -1.19 |
| BTC Hourly | nn | NN | 1021 | 449 | 572 | 43.98% | 40.83% | 40.83% | 6.02 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1021 | 447 | 574 | 43.78% | 40.42% | 42.29% | 6.22 pp | -127 | 53 | -2.40 |
| BTC Hourly | lstm | LSTM | 1021 | 429 | 592 | 42.02% | 34.58% | 39.38% | 7.98 pp | -163 | 53 | -3.08 |
| BTC Hourly | xgb | XGBoost | 1021 | 419 | 602 | 41.04% | 34.58% | 37.71% | 8.96 pp | -183 | 53 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 844 | 401 | 443 | 47.51% | 43.75% | 46.04% | 2.49 pp | -42 | 48 | -0.88 |
| BTC Daily | nn | NN | 844 | 394 | 450 | 46.68% | 45.83% | 45.42% | 3.32 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 844 | 389 | 455 | 46.09% | 37.92% | 44.38% | 3.91 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 844 | 356 | 488 | 42.18% | 35.83% | 39.79% | 7.82 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 844 | 351 | 493 | 41.59% | 37.50% | 41.04% | 8.41 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 854 | 337 | 517 | 39.46% | 37.92% | 36.25% | 10.54 pp | -180 | 48 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 616 | 297 | 319 | 48.21% | 47.08% | 47.29% | 1.79 pp | -22 | 57 | -0.39 |
| BTC Market Hours | nn | NN | 616 | 295 | 321 | 47.89% | 50.83% | 49.79% | 2.11 pp | -26 | 57 | -0.46 |
| BTC Market Hours | transformer | Transformer | 616 | 287 | 329 | 46.59% | 46.25% | 45.62% | 3.41 pp | -42 | 57 | -0.74 |
| BTC Market Hours | lstm | LSTM | 616 | 264 | 352 | 42.86% | 42.92% | 43.12% | 7.14 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 616 | 264 | 352 | 42.86% | 43.75% | 41.88% | 7.14 pp | -88 | 57 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 616 | 262 | 354 | 42.53% | 46.67% | 42.92% | 7.47 pp | -92 | 57 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 670 | 316 | 354 | 47.16% | 50.00% | 48.54% | 2.84 pp | -38 | 57 | -0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 670 | 313 | 357 | 46.72% | 48.75% | 47.08% | 3.28 pp | -44 | 57 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 670 | 311 | 359 | 46.42% | 47.92% | 47.92% | 3.58 pp | -48 | 57 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 670 | 278 | 392 | 41.49% | 43.33% | 41.67% | 8.51 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 670 | 275 | 395 | 41.04% | 44.17% | 41.25% | 8.96 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 670 | 275 | 395 | 41.04% | 43.75% | 41.25% | 8.96 pp | -120 | 57 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
