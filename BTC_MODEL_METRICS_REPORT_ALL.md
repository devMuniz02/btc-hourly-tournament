# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T01:03:38.654051+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1332 | 1044 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1208 | 843 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 967 | 605 | 361 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 969 | 659 | 308 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 246 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 605 | 290 | 315 | 47.93% | 45.83% | 47.29% | 2.07 pp | -25 | 56 | -0.45 |
| BTC Market Hours | nn | NN | 605 | 288 | 317 | 47.60% | 50.42% | 49.17% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 605 | 283 | 322 | 46.78% | 45.83% | 46.25% | 3.22 pp | -39 | 56 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 659 | 308 | 351 | 46.74% | 48.33% | 47.08% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | nn | NN | 659 | 308 | 351 | 46.74% | 48.33% | 47.71% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 833 | 397 | 436 | 47.66% | 44.58% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 659 | 306 | 353 | 46.43% | 47.92% | 47.50% | 3.57 pp | -47 | 56 | -0.84 |
| Consolidated Hourly | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1010 | 478 | 532 | 47.33% | 48.33% | 45.62% | 2.67 pp | -54 | 52 | -1.04 |
| BTC Daily | nn | NN | 833 | 388 | 445 | 46.58% | 45.00% | 45.21% | 3.42 pp | -57 | 48 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 1010 | 470 | 540 | 46.53% | 45.83% | 44.38% | 3.47 pp | -70 | 52 | -1.35 |
| BTC Daily | transformer | Transformer | 833 | 384 | 449 | 46.10% | 37.50% | 44.58% | 3.90 pp | -65 | 48 | -1.35 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 15 | -1.53 |
| BTC Market Hours | lstm | LSTM | 605 | 259 | 346 | 42.81% | 42.50% | 42.92% | 7.19 pp | -87 | 56 | -1.55 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 605 | 258 | 347 | 42.64% | 42.92% | 42.29% | 7.36 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 605 | 257 | 348 | 42.48% | 45.00% | 43.33% | 7.52 pp | -91 | 56 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 659 | 272 | 387 | 41.27% | 42.50% | 41.25% | 8.73 pp | -115 | 56 | -2.05 |
| Consolidated Hourly | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 659 | 269 | 390 | 40.82% | 42.92% | 40.42% | 9.18 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 659 | 269 | 390 | 40.82% | 42.50% | 40.62% | 9.18 pp | -121 | 56 | -2.16 |
| Consolidated Hourly | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 1010 | 445 | 565 | 44.06% | 42.50% | 41.04% | 5.94 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1010 | 445 | 565 | 44.06% | 41.67% | 43.33% | 5.94 pp | -120 | 52 | -2.31 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 833 | 352 | 481 | 42.26% | 35.83% | 40.21% | 7.74 pp | -129 | 48 | -2.69 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 35 | 54 | 39.33% | 39.33% | 39.33% | 10.67 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 833 | 345 | 488 | 41.42% | 36.67% | 40.62% | 8.58 pp | -143 | 48 | -2.98 |
| BTC Hourly | lstm | LSTM | 1010 | 427 | 583 | 42.28% | 36.25% | 39.79% | 7.72 pp | -156 | 52 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1010 | 415 | 595 | 41.09% | 35.42% | 38.33% | 8.91 pp | -180 | 52 | -3.46 |
| BTC Daily | xgb | XGBoost | 843 | 332 | 511 | 39.38% | 37.08% | 36.67% | 10.62 pp | -179 | 48 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1010 | 478 | 532 | 47.33% | 48.33% | 45.62% | 2.67 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1010 | 470 | 540 | 46.53% | 45.83% | 44.38% | 3.47 pp | -70 | 52 | -1.35 |
| BTC Hourly | nn | NN | 1010 | 445 | 565 | 44.06% | 42.50% | 41.04% | 5.94 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1010 | 445 | 565 | 44.06% | 41.67% | 43.33% | 5.94 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 1010 | 427 | 583 | 42.28% | 36.25% | 39.79% | 7.72 pp | -156 | 52 | -3.00 |
| BTC Hourly | xgb | XGBoost | 1010 | 415 | 595 | 41.09% | 35.42% | 38.33% | 8.91 pp | -180 | 52 | -3.46 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 833 | 397 | 436 | 47.66% | 44.58% | 45.83% | 2.34 pp | -39 | 48 | -0.81 |
| BTC Daily | nn | NN | 833 | 388 | 445 | 46.58% | 45.00% | 45.21% | 3.42 pp | -57 | 48 | -1.19 |
| BTC Daily | transformer | Transformer | 833 | 384 | 449 | 46.10% | 37.50% | 44.58% | 3.90 pp | -65 | 48 | -1.35 |
| BTC Daily | lstm | LSTM | 833 | 352 | 481 | 42.26% | 35.83% | 40.21% | 7.74 pp | -129 | 48 | -2.69 |
| BTC Daily | rf | RandomForest | 833 | 345 | 488 | 41.42% | 36.67% | 40.62% | 8.58 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 843 | 332 | 511 | 39.38% | 37.08% | 36.67% | 10.62 pp | -179 | 48 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 605 | 290 | 315 | 47.93% | 45.83% | 47.29% | 2.07 pp | -25 | 56 | -0.45 |
| BTC Market Hours | nn | NN | 605 | 288 | 317 | 47.60% | 50.42% | 49.17% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 605 | 283 | 322 | 46.78% | 45.83% | 46.25% | 3.22 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 605 | 259 | 346 | 42.81% | 42.50% | 42.92% | 7.19 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 605 | 258 | 347 | 42.64% | 42.92% | 42.29% | 7.36 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 605 | 257 | 348 | 42.48% | 45.00% | 43.33% | 7.52 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 659 | 308 | 351 | 46.74% | 48.33% | 47.08% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | nn | NN | 659 | 308 | 351 | 46.74% | 48.33% | 47.71% | 3.26 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 659 | 306 | 353 | 46.43% | 47.92% | 47.50% | 3.57 pp | -47 | 56 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 659 | 272 | 387 | 41.27% | 42.50% | 41.25% | 8.73 pp | -115 | 56 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 659 | 269 | 390 | 40.82% | 42.92% | 40.42% | 9.18 pp | -121 | 56 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 659 | 269 | 390 | 40.82% | 42.50% | 40.62% | 9.18 pp | -121 | 56 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 35 | 54 | 39.33% | 39.33% | 39.33% | 10.67 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
