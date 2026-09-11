# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T19:02:02.864359+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1344 | 1056 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1220 | 855 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 18:00:00+00:00 | 986 | 617 | 368 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 18:00:00+00:00 | 988 | 671 | 315 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 255 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 255 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 255 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 256 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 617 | 297 | 320 | 48.14% | 46.67% | 47.29% | 1.86 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 617 | 295 | 322 | 47.81% | 50.42% | 49.58% | 2.19 pp | -27 | 57 | -0.47 |
| BTC Market Hours Daily | nn | NN | 671 | 317 | 354 | 47.24% | 50.42% | 48.75% | 2.76 pp | -37 | 57 | -0.65 |
| BTC Market Hours | transformer | Transformer | 617 | 288 | 329 | 46.68% | 46.25% | 45.62% | 3.32 pp | -41 | 57 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 671 | 314 | 357 | 46.80% | 49.17% | 47.29% | 3.20 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 671 | 311 | 360 | 46.35% | 47.92% | 47.71% | 3.65 pp | -49 | 57 | -0.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 845 | 401 | 444 | 47.46% | 43.33% | 46.04% | 2.54 pp | -43 | 48 | -0.90 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1022 | 483 | 539 | 47.26% | 46.67% | 46.04% | 2.74 pp | -56 | 53 | -1.06 |
| BTC Daily | nn | NN | 845 | 394 | 451 | 46.63% | 45.83% | 45.21% | 3.37 pp | -57 | 48 | -1.19 |
| Consolidated Hourly | rf | RandomForest | 255 | 118 | 137 | 46.27% | 46.25% | 46.27% | 3.73 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 118 | 137 | 46.27% | 46.25% | 46.27% | 3.73 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1022 | 479 | 543 | 46.87% | 46.67% | 44.79% | 3.13 pp | -64 | 53 | -1.21 |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Daily | transformer | Transformer | 845 | 389 | 456 | 46.04% | 37.92% | 44.38% | 3.96 pp | -67 | 48 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 45.00% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 45.00% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 617 | 265 | 352 | 42.95% | 43.33% | 43.33% | 7.05 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 617 | 264 | 353 | 42.79% | 43.33% | 41.88% | 7.21 pp | -89 | 57 | -1.56 |
| Consolidated Hourly | lstm | LSTM | 255 | 115 | 140 | 45.10% | 44.58% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 115 | 140 | 45.10% | 44.58% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 41 | 54 | 43.16% | 43.16% | 43.16% | 6.84 pp | -13 | 8 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 617 | 262 | 355 | 42.46% | 46.25% | 42.71% | 7.54 pp | -93 | 57 | -1.63 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 255 | 113 | 142 | 44.31% | 43.75% | 44.31% | 5.69 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 113 | 142 | 44.31% | 43.75% | 44.31% | 5.69 pp | -29 | 16 | -1.81 |
| BTC Market Hours Daily | rf | RandomForest | 671 | 279 | 392 | 41.58% | 43.75% | 41.67% | 8.42 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 671 | 275 | 396 | 40.98% | 43.75% | 41.04% | 9.02 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 671 | 275 | 396 | 40.98% | 43.75% | 41.25% | 9.02 pp | -121 | 57 | -2.12 |
| Consolidated Hourly | nn | NN | 255 | 109 | 146 | 42.75% | 43.33% | 42.75% | 7.25 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 109 | 146 | 42.75% | 43.33% | 42.75% | 7.25 pp | -37 | 16 | -2.31 |
| BTC Hourly | nn | NN | 1022 | 449 | 573 | 43.93% | 40.83% | 40.62% | 6.07 pp | -124 | 53 | -2.34 |
| BTC Hourly | rf | RandomForest | 1022 | 447 | 575 | 43.74% | 40.42% | 42.08% | 6.26 pp | -128 | 53 | -2.42 |
| Consolidated Hourly | xgb | XGBoost | 255 | 107 | 148 | 41.96% | 41.25% | 41.96% | 8.04 pp | -41 | 16 | -2.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 107 | 148 | 41.96% | 41.25% | 41.96% | 8.04 pp | -41 | 16 | -2.56 |
| Consolidated Market Hours Daily | nn | NN | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 845 | 357 | 488 | 42.25% | 36.25% | 40.00% | 7.75 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 845 | 351 | 494 | 41.54% | 37.08% | 40.83% | 8.46 pp | -143 | 48 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 1022 | 429 | 593 | 41.98% | 34.58% | 39.17% | 8.02 pp | -164 | 53 | -3.09 |
| BTC Hourly | xgb | XGBoost | 1022 | 419 | 603 | 41.00% | 34.58% | 37.50% | 9.00 pp | -184 | 53 | -3.47 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 33 | 62 | 34.74% | 34.74% | 34.74% | 15.26 pp | -29 | 8 | -3.62 |
| BTC Daily | xgb | XGBoost | 855 | 338 | 517 | 39.53% | 37.92% | 36.46% | 10.47 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1022 | 483 | 539 | 47.26% | 46.67% | 46.04% | 2.74 pp | -56 | 53 | -1.06 |
| BTC Hourly | transformer | Transformer | 1022 | 479 | 543 | 46.87% | 46.67% | 44.79% | 3.13 pp | -64 | 53 | -1.21 |
| BTC Hourly | nn | NN | 1022 | 449 | 573 | 43.93% | 40.83% | 40.62% | 6.07 pp | -124 | 53 | -2.34 |
| BTC Hourly | rf | RandomForest | 1022 | 447 | 575 | 43.74% | 40.42% | 42.08% | 6.26 pp | -128 | 53 | -2.42 |
| BTC Hourly | lstm | LSTM | 1022 | 429 | 593 | 41.98% | 34.58% | 39.17% | 8.02 pp | -164 | 53 | -3.09 |
| BTC Hourly | xgb | XGBoost | 1022 | 419 | 603 | 41.00% | 34.58% | 37.50% | 9.00 pp | -184 | 53 | -3.47 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 845 | 401 | 444 | 47.46% | 43.33% | 46.04% | 2.54 pp | -43 | 48 | -0.90 |
| BTC Daily | nn | NN | 845 | 394 | 451 | 46.63% | 45.83% | 45.21% | 3.37 pp | -57 | 48 | -1.19 |
| BTC Daily | transformer | Transformer | 845 | 389 | 456 | 46.04% | 37.92% | 44.38% | 3.96 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 845 | 357 | 488 | 42.25% | 36.25% | 40.00% | 7.75 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 845 | 351 | 494 | 41.54% | 37.08% | 40.83% | 8.46 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 855 | 338 | 517 | 39.53% | 37.92% | 36.46% | 10.47 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 617 | 297 | 320 | 48.14% | 46.67% | 47.29% | 1.86 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 617 | 295 | 322 | 47.81% | 50.42% | 49.58% | 2.19 pp | -27 | 57 | -0.47 |
| BTC Market Hours | transformer | Transformer | 617 | 288 | 329 | 46.68% | 46.25% | 45.62% | 3.32 pp | -41 | 57 | -0.72 |
| BTC Market Hours | lstm | LSTM | 617 | 265 | 352 | 42.95% | 43.33% | 43.33% | 7.05 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 617 | 264 | 353 | 42.79% | 43.33% | 41.88% | 7.21 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 617 | 262 | 355 | 42.46% | 46.25% | 42.71% | 7.54 pp | -93 | 57 | -1.63 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 671 | 317 | 354 | 47.24% | 50.42% | 48.75% | 2.76 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 671 | 314 | 357 | 46.80% | 49.17% | 47.29% | 3.20 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 671 | 311 | 360 | 46.35% | 47.92% | 47.71% | 3.65 pp | -49 | 57 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 671 | 279 | 392 | 41.58% | 43.75% | 41.67% | 8.42 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 671 | 275 | 396 | 40.98% | 43.75% | 41.04% | 9.02 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 671 | 275 | 396 | 40.98% | 43.75% | 41.25% | 9.02 pp | -121 | 57 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 255 | 118 | 137 | 46.27% | 46.25% | 46.27% | 3.73 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 45.00% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 255 | 115 | 140 | 45.10% | 44.58% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 255 | 113 | 142 | 44.31% | 43.75% | 44.31% | 5.69 pp | -29 | 16 | -1.81 |
| Consolidated Hourly | nn | NN | 255 | 109 | 146 | 42.75% | 43.33% | 42.75% | 7.25 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 255 | 107 | 148 | 41.96% | 41.25% | 41.96% | 8.04 pp | -41 | 16 | -2.56 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 118 | 137 | 46.27% | 46.25% | 46.27% | 3.73 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 45.00% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 115 | 140 | 45.10% | 44.58% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 113 | 142 | 44.31% | 43.75% | 44.31% | 5.69 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 109 | 146 | 42.75% | 43.33% | 42.75% | 7.25 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 107 | 148 | 41.96% | 41.25% | 41.96% | 8.04 pp | -41 | 16 | -2.56 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 41 | 54 | 43.16% | 43.16% | 43.16% | 6.84 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours Daily | nn | NN | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 33 | 62 | 34.74% | 34.74% | 34.74% | 15.26 pp | -29 | 8 | -3.62 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
