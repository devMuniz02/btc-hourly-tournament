# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T03:23:04.383404+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1333 | 1045 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1209 | 844 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 968 | 606 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 970 | 660 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 606 | 291 | 315 | 48.02% | 46.25% | 47.50% | 1.98 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 606 | 288 | 318 | 47.52% | 50.00% | 48.96% | 2.48 pp | -30 | 56 | -0.54 |
| BTC Market Hours | transformer | Transformer | 606 | 283 | 323 | 46.70% | 45.42% | 46.04% | 3.30 pp | -40 | 56 | -0.71 |
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 660 | 309 | 351 | 46.82% | 48.75% | 47.08% | 3.18 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | nn | NN | 660 | 309 | 351 | 46.82% | 48.75% | 47.71% | 3.18 pp | -42 | 56 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 834 | 398 | 436 | 47.72% | 45.00% | 45.83% | 2.28 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 660 | 307 | 353 | 46.52% | 48.33% | 47.71% | 3.48 pp | -46 | 56 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1011 | 479 | 532 | 47.38% | 48.33% | 45.83% | 2.62 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 834 | 388 | 446 | 46.52% | 45.00% | 45.21% | 3.48 pp | -58 | 48 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1011 | 472 | 539 | 46.69% | 45.83% | 44.58% | 3.31 pp | -67 | 52 | -1.29 |
| BTC Daily | transformer | Transformer | 834 | 384 | 450 | 46.04% | 37.50% | 44.38% | 3.96 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 606 | 259 | 347 | 42.74% | 42.50% | 42.92% | 7.26 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 606 | 259 | 347 | 42.74% | 42.92% | 42.50% | 7.26 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 606 | 257 | 349 | 42.41% | 45.00% | 43.33% | 7.59 pp | -92 | 56 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 660 | 273 | 387 | 41.36% | 42.92% | 41.25% | 8.64 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 660 | 270 | 390 | 40.91% | 43.33% | 40.62% | 9.09 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 660 | 270 | 390 | 40.91% | 42.50% | 40.62% | 9.09 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1011 | 447 | 564 | 44.21% | 42.92% | 41.25% | 5.79 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1011 | 445 | 566 | 44.02% | 40.83% | 43.12% | 5.98 pp | -121 | 52 | -2.33 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| BTC Daily | lstm | LSTM | 834 | 353 | 481 | 42.33% | 36.25% | 40.21% | 7.67 pp | -128 | 48 | -2.67 |
| Consolidated Hourly | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| BTC Hourly | lstm | LSTM | 1011 | 428 | 583 | 42.33% | 36.67% | 39.79% | 7.67 pp | -155 | 52 | -2.98 |
| BTC Daily | rf | RandomForest | 834 | 345 | 489 | 41.37% | 36.67% | 40.62% | 8.63 pp | -144 | 48 | -3.00 |
| Consolidated Hourly | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1011 | 415 | 596 | 41.05% | 34.58% | 38.12% | 8.95 pp | -181 | 52 | -3.48 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 844 | 332 | 512 | 39.34% | 37.08% | 36.67% | 10.66 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1011 | 479 | 532 | 47.38% | 48.33% | 45.83% | 2.62 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1011 | 472 | 539 | 46.69% | 45.83% | 44.58% | 3.31 pp | -67 | 52 | -1.29 |
| BTC Hourly | nn | NN | 1011 | 447 | 564 | 44.21% | 42.92% | 41.25% | 5.79 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1011 | 445 | 566 | 44.02% | 40.83% | 43.12% | 5.98 pp | -121 | 52 | -2.33 |
| BTC Hourly | lstm | LSTM | 1011 | 428 | 583 | 42.33% | 36.67% | 39.79% | 7.67 pp | -155 | 52 | -2.98 |
| BTC Hourly | xgb | XGBoost | 1011 | 415 | 596 | 41.05% | 34.58% | 38.12% | 8.95 pp | -181 | 52 | -3.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 834 | 398 | 436 | 47.72% | 45.00% | 45.83% | 2.28 pp | -38 | 48 | -0.79 |
| BTC Daily | nn | NN | 834 | 388 | 446 | 46.52% | 45.00% | 45.21% | 3.48 pp | -58 | 48 | -1.21 |
| BTC Daily | transformer | Transformer | 834 | 384 | 450 | 46.04% | 37.50% | 44.38% | 3.96 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 834 | 353 | 481 | 42.33% | 36.25% | 40.21% | 7.67 pp | -128 | 48 | -2.67 |
| BTC Daily | rf | RandomForest | 834 | 345 | 489 | 41.37% | 36.67% | 40.62% | 8.63 pp | -144 | 48 | -3.00 |
| BTC Daily | xgb | XGBoost | 844 | 332 | 512 | 39.34% | 37.08% | 36.67% | 10.66 pp | -180 | 48 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 606 | 291 | 315 | 48.02% | 46.25% | 47.50% | 1.98 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 606 | 288 | 318 | 47.52% | 50.00% | 48.96% | 2.48 pp | -30 | 56 | -0.54 |
| BTC Market Hours | transformer | Transformer | 606 | 283 | 323 | 46.70% | 45.42% | 46.04% | 3.30 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 606 | 259 | 347 | 42.74% | 42.50% | 42.92% | 7.26 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 606 | 259 | 347 | 42.74% | 42.92% | 42.50% | 7.26 pp | -88 | 56 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 606 | 257 | 349 | 42.41% | 45.00% | 43.33% | 7.59 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 660 | 309 | 351 | 46.82% | 48.75% | 47.08% | 3.18 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | nn | NN | 660 | 309 | 351 | 46.82% | 48.75% | 47.71% | 3.18 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 660 | 307 | 353 | 46.52% | 48.33% | 47.71% | 3.48 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 660 | 273 | 387 | 41.36% | 42.92% | 41.25% | 8.64 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 660 | 270 | 390 | 40.91% | 43.33% | 40.62% | 9.09 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 660 | 270 | 390 | 40.91% | 42.50% | 40.62% | 9.09 pp | -120 | 56 | -2.14 |

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
