# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T09:39:41.778150+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1290 | 1002 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1166 | 801 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 886 | 563 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 888 | 617 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 205 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T12:00:00+00:00 | 206 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 563 | 274 | 289 | 48.67% | 47.08% | 47.71% | 1.33 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 563 | 267 | 296 | 47.42% | 50.83% | 48.96% | 2.58 pp | -29 | 53 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 791 | 382 | 409 | 48.29% | 47.08% | 48.12% | 1.71 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 563 | 265 | 298 | 47.07% | 46.67% | 47.29% | 2.93 pp | -33 | 53 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 617 | 288 | 329 | 46.68% | 48.75% | 47.50% | 3.32 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 617 | 287 | 330 | 46.52% | 49.17% | 47.29% | 3.48 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | nn | NN | 617 | 287 | 330 | 46.52% | 47.08% | 47.92% | 3.48 pp | -43 | 53 | -0.81 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 968 | 460 | 508 | 47.52% | 49.17% | 46.25% | 2.48 pp | -48 | 50 | -0.96 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 791 | 368 | 423 | 46.52% | 45.42% | 45.21% | 3.48 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 791 | 368 | 423 | 46.52% | 40.83% | 46.67% | 3.48 pp | -55 | 46 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| BTC Hourly | transformer | Transformer | 968 | 451 | 517 | 46.59% | 44.17% | 43.75% | 3.41 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| BTC Market Hours | lstm | LSTM | 563 | 245 | 318 | 43.52% | 42.50% | 43.96% | 6.48 pp | -73 | 53 | -1.38 |
| BTC Market Hours | rf | RandomForest | 563 | 243 | 320 | 43.16% | 45.42% | 43.54% | 6.84 pp | -77 | 53 | -1.45 |
| Consolidated Hourly | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 563 | 240 | 323 | 42.63% | 45.83% | 43.12% | 7.37 pp | -83 | 53 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 617 | 258 | 359 | 41.82% | 44.17% | 40.83% | 8.18 pp | -101 | 53 | -1.91 |
| Consolidated Hourly | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 617 | 252 | 365 | 40.84% | 40.83% | 40.42% | 9.16 pp | -113 | 53 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 617 | 252 | 365 | 40.84% | 43.33% | 40.62% | 9.16 pp | -113 | 53 | -2.13 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Hourly | rf | RandomForest | 968 | 428 | 540 | 44.21% | 41.67% | 42.50% | 5.79 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 968 | 427 | 541 | 44.11% | 41.25% | 42.50% | 5.89 pp | -114 | 50 | -2.28 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 791 | 333 | 458 | 42.10% | 33.75% | 40.21% | 7.90 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 791 | 331 | 460 | 41.85% | 38.33% | 41.67% | 8.15 pp | -129 | 46 | -2.80 |
| BTC Hourly | lstm | LSTM | 968 | 412 | 556 | 42.56% | 37.50% | 41.46% | 7.44 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 968 | 399 | 569 | 41.22% | 36.25% | 38.54% | 8.78 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 801 | 314 | 487 | 39.20% | 36.25% | 36.25% | 10.80 pp | -173 | 46 | -3.76 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 968 | 460 | 508 | 47.52% | 49.17% | 46.25% | 2.48 pp | -48 | 50 | -0.96 |
| BTC Hourly | transformer | Transformer | 968 | 451 | 517 | 46.59% | 44.17% | 43.75% | 3.41 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 968 | 428 | 540 | 44.21% | 41.67% | 42.50% | 5.79 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 968 | 427 | 541 | 44.11% | 41.25% | 42.50% | 5.89 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 968 | 412 | 556 | 42.56% | 37.50% | 41.46% | 7.44 pp | -144 | 50 | -2.88 |
| BTC Hourly | xgb | XGBoost | 968 | 399 | 569 | 41.22% | 36.25% | 38.54% | 8.78 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 791 | 382 | 409 | 48.29% | 47.08% | 48.12% | 1.71 pp | -27 | 46 | -0.59 |
| BTC Daily | nn | NN | 791 | 368 | 423 | 46.52% | 45.42% | 45.21% | 3.48 pp | -55 | 46 | -1.20 |
| BTC Daily | transformer | Transformer | 791 | 368 | 423 | 46.52% | 40.83% | 46.67% | 3.48 pp | -55 | 46 | -1.20 |
| BTC Daily | lstm | LSTM | 791 | 333 | 458 | 42.10% | 33.75% | 40.21% | 7.90 pp | -125 | 46 | -2.72 |
| BTC Daily | rf | RandomForest | 791 | 331 | 460 | 41.85% | 38.33% | 41.67% | 8.15 pp | -129 | 46 | -2.80 |
| BTC Daily | xgb | XGBoost | 801 | 314 | 487 | 39.20% | 36.25% | 36.25% | 10.80 pp | -173 | 46 | -3.76 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 563 | 274 | 289 | 48.67% | 47.08% | 47.71% | 1.33 pp | -15 | 53 | -0.28 |
| BTC Market Hours | nn | NN | 563 | 267 | 296 | 47.42% | 50.83% | 48.96% | 2.58 pp | -29 | 53 | -0.55 |
| BTC Market Hours | transformer | Transformer | 563 | 265 | 298 | 47.07% | 46.67% | 47.29% | 2.93 pp | -33 | 53 | -0.62 |
| BTC Market Hours | lstm | LSTM | 563 | 245 | 318 | 43.52% | 42.50% | 43.96% | 6.48 pp | -73 | 53 | -1.38 |
| BTC Market Hours | rf | RandomForest | 563 | 243 | 320 | 43.16% | 45.42% | 43.54% | 6.84 pp | -77 | 53 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 563 | 240 | 323 | 42.63% | 45.83% | 43.12% | 7.37 pp | -83 | 53 | -1.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 617 | 288 | 329 | 46.68% | 48.75% | 47.50% | 3.32 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 617 | 287 | 330 | 46.52% | 49.17% | 47.29% | 3.48 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | nn | NN | 617 | 287 | 330 | 46.52% | 47.08% | 47.92% | 3.48 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 617 | 258 | 359 | 41.82% | 44.17% | 40.83% | 8.18 pp | -101 | 53 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 617 | 252 | 365 | 40.84% | 40.83% | 40.42% | 9.16 pp | -113 | 53 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 617 | 252 | 365 | 40.84% | 43.33% | 40.62% | 9.16 pp | -113 | 53 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 101 | 104 | 49.27% | 49.27% | 49.27% | 0.73 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 94 | 111 | 45.85% | 45.85% | 45.85% | 4.15 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
