# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T04:06:00.944374+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1221 | 933 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1097 | 732 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 765 | 494 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 767 | 548 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 143 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T17:00:00+00:00 | 144 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 494 | 238 | 256 | 48.18% | 44.17% | 47.92% | 1.82 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 494 | 233 | 261 | 47.17% | 50.00% | 47.71% | 2.83 pp | -28 | 47 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 722 | 348 | 374 | 48.20% | 46.67% | 47.92% | 1.80 pp | -26 | 43 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 494 | 231 | 263 | 46.76% | 43.33% | 47.29% | 3.24 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 722 | 346 | 376 | 47.92% | 47.08% | 50.42% | 2.08 pp | -30 | 43 | -0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 548 | 254 | 294 | 46.35% | 48.75% | 47.08% | 3.65 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 548 | 253 | 295 | 46.17% | 45.00% | 47.29% | 3.83 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 548 | 251 | 297 | 45.80% | 48.33% | 46.88% | 4.20 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 722 | 335 | 387 | 46.40% | 44.58% | 47.71% | 3.60 pp | -52 | 43 | -1.21 |
| Consolidated Hourly | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| BTC Market Hours | rf | RandomForest | 494 | 214 | 280 | 43.32% | 44.17% | 43.54% | 6.68 pp | -66 | 47 | -1.40 |
| BTC Market Hours | lstm | LSTM | 494 | 213 | 281 | 43.12% | 40.83% | 43.12% | 6.88 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 494 | 203 | 291 | 41.09% | 40.83% | 41.04% | 8.91 pp | -88 | 47 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 548 | 228 | 320 | 41.61% | 42.08% | 41.25% | 8.39 pp | -92 | 47 | -1.96 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 548 | 221 | 327 | 40.33% | 38.33% | 40.83% | 9.67 pp | -106 | 47 | -2.26 |
| Consolidated Hourly | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |
| BTC Daily | lstm | LSTM | 722 | 312 | 410 | 43.21% | 37.92% | 42.08% | 6.79 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 548 | 220 | 328 | 40.15% | 41.25% | 39.38% | 9.85 pp | -108 | 47 | -2.30 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 722 | 309 | 413 | 42.80% | 41.25% | 43.54% | 7.20 pp | -104 | 43 | -2.42 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 732 | 291 | 441 | 39.75% | 36.67% | 38.75% | 10.25 pp | -150 | 43 | -3.49 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 722 | 348 | 374 | 48.20% | 46.67% | 47.92% | 1.80 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 722 | 346 | 376 | 47.92% | 47.08% | 50.42% | 2.08 pp | -30 | 43 | -0.70 |
| BTC Daily | nn | NN | 722 | 335 | 387 | 46.40% | 44.58% | 47.71% | 3.60 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 722 | 312 | 410 | 43.21% | 37.92% | 42.08% | 6.79 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 722 | 309 | 413 | 42.80% | 41.25% | 43.54% | 7.20 pp | -104 | 43 | -2.42 |
| BTC Daily | xgb | XGBoost | 732 | 291 | 441 | 39.75% | 36.67% | 38.75% | 10.25 pp | -150 | 43 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 494 | 238 | 256 | 48.18% | 44.17% | 47.92% | 1.82 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 494 | 233 | 261 | 47.17% | 50.00% | 47.71% | 2.83 pp | -28 | 47 | -0.60 |
| BTC Market Hours | transformer | Transformer | 494 | 231 | 263 | 46.76% | 43.33% | 47.29% | 3.24 pp | -32 | 47 | -0.68 |
| BTC Market Hours | rf | RandomForest | 494 | 214 | 280 | 43.32% | 44.17% | 43.54% | 6.68 pp | -66 | 47 | -1.40 |
| BTC Market Hours | lstm | LSTM | 494 | 213 | 281 | 43.12% | 40.83% | 43.12% | 6.88 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 494 | 203 | 291 | 41.09% | 40.83% | 41.04% | 8.91 pp | -88 | 47 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 548 | 254 | 294 | 46.35% | 48.75% | 47.08% | 3.65 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 548 | 253 | 295 | 46.17% | 45.00% | 47.29% | 3.83 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 548 | 251 | 297 | 45.80% | 48.33% | 46.88% | 4.20 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 548 | 228 | 320 | 41.61% | 42.08% | 41.25% | 8.39 pp | -92 | 47 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 548 | 221 | 327 | 40.33% | 38.33% | 40.83% | 9.67 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 548 | 220 | 328 | 40.15% | 41.25% | 39.38% | 9.85 pp | -108 | 47 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 73 | 70 | 51.05% | 51.05% | 51.05% | 1.05 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 70 | 73 | 48.95% | 48.95% | 48.95% | 1.05 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 65 | 78 | 45.45% | 45.45% | 45.45% | 4.55 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
