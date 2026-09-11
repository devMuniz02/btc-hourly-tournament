# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T05:19:07.463892+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1335 | 1047 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1211 | 846 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 970 | 608 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 971 | 661 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 608 | 293 | 315 | 48.19% | 46.67% | 47.92% | 1.81 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 608 | 290 | 318 | 47.70% | 50.00% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 608 | 285 | 323 | 46.88% | 46.25% | 46.04% | 3.12 pp | -38 | 56 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 661 | 310 | 351 | 46.90% | 49.17% | 47.08% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | nn | NN | 661 | 310 | 351 | 46.90% | 48.75% | 47.71% | 3.10 pp | -41 | 56 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 836 | 398 | 438 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 661 | 307 | 354 | 46.44% | 48.33% | 47.50% | 3.56 pp | -47 | 56 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1013 | 479 | 534 | 47.29% | 47.92% | 45.62% | 2.71 pp | -55 | 52 | -1.06 |
| BTC Daily | nn | NN | 836 | 389 | 447 | 46.53% | 45.00% | 45.21% | 3.47 pp | -58 | 48 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1013 | 472 | 541 | 46.59% | 45.83% | 44.38% | 3.41 pp | -69 | 52 | -1.33 |
| BTC Daily | transformer | Transformer | 836 | 386 | 450 | 46.17% | 38.33% | 44.58% | 3.83 pp | -64 | 48 | -1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| BTC Market Hours | lstm | LSTM | 608 | 261 | 347 | 42.93% | 43.33% | 43.33% | 7.07 pp | -86 | 56 | -1.54 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 608 | 260 | 348 | 42.76% | 43.33% | 42.29% | 7.24 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 608 | 258 | 350 | 42.43% | 45.42% | 43.33% | 7.57 pp | -92 | 56 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 661 | 274 | 387 | 41.45% | 42.92% | 41.46% | 8.55 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 661 | 270 | 391 | 40.85% | 43.33% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 661 | 270 | 391 | 40.85% | 42.50% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| BTC Hourly | nn | NN | 1013 | 447 | 566 | 44.13% | 42.50% | 41.04% | 5.87 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1013 | 445 | 568 | 43.93% | 40.42% | 42.71% | 6.07 pp | -123 | 52 | -2.37 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| BTC Daily | lstm | LSTM | 836 | 353 | 483 | 42.22% | 35.83% | 40.00% | 7.78 pp | -130 | 48 | -2.71 |
| Consolidated Hourly | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| BTC Daily | rf | RandomForest | 836 | 347 | 489 | 41.51% | 37.08% | 40.83% | 8.49 pp | -142 | 48 | -2.96 |
| BTC Hourly | lstm | LSTM | 1013 | 428 | 585 | 42.25% | 36.25% | 39.58% | 7.75 pp | -157 | 52 | -3.02 |
| Consolidated Hourly | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1013 | 415 | 598 | 40.97% | 34.58% | 37.92% | 9.03 pp | -183 | 52 | -3.52 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 846 | 334 | 512 | 39.48% | 37.50% | 36.67% | 10.52 pp | -178 | 48 | -3.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1013 | 479 | 534 | 47.29% | 47.92% | 45.62% | 2.71 pp | -55 | 52 | -1.06 |
| BTC Hourly | transformer | Transformer | 1013 | 472 | 541 | 46.59% | 45.83% | 44.38% | 3.41 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1013 | 447 | 566 | 44.13% | 42.50% | 41.04% | 5.87 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1013 | 445 | 568 | 43.93% | 40.42% | 42.71% | 6.07 pp | -123 | 52 | -2.37 |
| BTC Hourly | lstm | LSTM | 1013 | 428 | 585 | 42.25% | 36.25% | 39.58% | 7.75 pp | -157 | 52 | -3.02 |
| BTC Hourly | xgb | XGBoost | 1013 | 415 | 598 | 40.97% | 34.58% | 37.92% | 9.03 pp | -183 | 52 | -3.52 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 836 | 398 | 438 | 47.61% | 44.58% | 45.83% | 2.39 pp | -40 | 48 | -0.83 |
| BTC Daily | nn | NN | 836 | 389 | 447 | 46.53% | 45.00% | 45.21% | 3.47 pp | -58 | 48 | -1.21 |
| BTC Daily | transformer | Transformer | 836 | 386 | 450 | 46.17% | 38.33% | 44.58% | 3.83 pp | -64 | 48 | -1.33 |
| BTC Daily | lstm | LSTM | 836 | 353 | 483 | 42.22% | 35.83% | 40.00% | 7.78 pp | -130 | 48 | -2.71 |
| BTC Daily | rf | RandomForest | 836 | 347 | 489 | 41.51% | 37.08% | 40.83% | 8.49 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 846 | 334 | 512 | 39.48% | 37.50% | 36.67% | 10.52 pp | -178 | 48 | -3.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 608 | 293 | 315 | 48.19% | 46.67% | 47.92% | 1.81 pp | -22 | 56 | -0.39 |
| BTC Market Hours | nn | NN | 608 | 290 | 318 | 47.70% | 50.00% | 49.38% | 2.30 pp | -28 | 56 | -0.50 |
| BTC Market Hours | transformer | Transformer | 608 | 285 | 323 | 46.88% | 46.25% | 46.04% | 3.12 pp | -38 | 56 | -0.68 |
| BTC Market Hours | lstm | LSTM | 608 | 261 | 347 | 42.93% | 43.33% | 43.33% | 7.07 pp | -86 | 56 | -1.54 |
| BTC Market Hours | rf | RandomForest | 608 | 260 | 348 | 42.76% | 43.33% | 42.29% | 7.24 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 608 | 258 | 350 | 42.43% | 45.42% | 43.33% | 7.57 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 661 | 310 | 351 | 46.90% | 49.17% | 47.08% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | nn | NN | 661 | 310 | 351 | 46.90% | 48.75% | 47.71% | 3.10 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 661 | 307 | 354 | 46.44% | 48.33% | 47.50% | 3.56 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 661 | 274 | 387 | 41.45% | 42.92% | 41.46% | 8.55 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 661 | 270 | 391 | 40.85% | 43.33% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 661 | 270 | 391 | 40.85% | 42.50% | 40.62% | 9.15 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
