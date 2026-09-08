# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T00:19:09.000376+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1284 | 996 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1160 | 795 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 879 | 557 | 321 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 881 | 611 | 268 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 202 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 47.50% | 47.92% | 1.17 pp | -13 | 52 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 557 | 265 | 292 | 47.58% | 51.25% | 49.38% | 2.42 pp | -27 | 52 | -0.52 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 785 | 379 | 406 | 48.28% | 46.67% | 47.92% | 1.72 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 557 | 263 | 294 | 47.22% | 47.08% | 47.71% | 2.78 pp | -31 | 52 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 611 | 287 | 324 | 46.97% | 50.00% | 47.92% | 3.03 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 611 | 286 | 325 | 46.81% | 48.33% | 48.33% | 3.19 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 611 | 285 | 326 | 46.64% | 49.58% | 47.50% | 3.36 pp | -41 | 52 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 962 | 458 | 504 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 30 | 35 | 46.15% | 46.15% | 46.15% | 3.85 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 785 | 364 | 421 | 46.37% | 40.00% | 46.25% | 3.63 pp | -57 | 46 | -1.24 |
| BTC Daily | nn | NN | 785 | 363 | 422 | 46.24% | 44.58% | 45.00% | 3.76 pp | -59 | 46 | -1.28 |
| BTC Hourly | transformer | Transformer | 962 | 447 | 515 | 46.47% | 44.58% | 43.75% | 3.53 pp | -68 | 50 | -1.36 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 557 | 241 | 316 | 43.27% | 41.67% | 43.75% | 6.73 pp | -75 | 52 | -1.44 |
| BTC Market Hours | rf | RandomForest | 557 | 241 | 316 | 43.27% | 45.83% | 43.33% | 6.73 pp | -75 | 52 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 557 | 236 | 321 | 42.37% | 45.00% | 42.71% | 7.63 pp | -85 | 52 | -1.63 |
| BTC Market Hours Daily | rf | RandomForest | 611 | 256 | 355 | 41.90% | 44.17% | 41.04% | 8.10 pp | -99 | 52 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 611 | 248 | 363 | 40.59% | 40.42% | 40.42% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 611 | 248 | 363 | 40.59% | 42.50% | 40.00% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Hourly | nn | NN | 962 | 425 | 537 | 44.18% | 41.67% | 42.50% | 5.82 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 962 | 425 | 537 | 44.18% | 42.08% | 42.71% | 5.82 pp | -112 | 50 | -2.24 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 785 | 329 | 456 | 41.91% | 33.33% | 39.58% | 8.09 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 785 | 329 | 456 | 41.91% | 38.33% | 41.67% | 8.09 pp | -127 | 46 | -2.76 |
| BTC Hourly | lstm | LSTM | 962 | 410 | 552 | 42.62% | 37.08% | 41.67% | 7.38 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 962 | 397 | 565 | 41.27% | 37.08% | 38.96% | 8.73 pp | -168 | 50 | -3.36 |
| BTC Daily | xgb | XGBoost | 795 | 310 | 485 | 38.99% | 35.42% | 36.46% | 11.01 pp | -175 | 46 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 962 | 458 | 504 | 47.61% | 49.58% | 46.88% | 2.39 pp | -46 | 50 | -0.92 |
| BTC Hourly | transformer | Transformer | 962 | 447 | 515 | 46.47% | 44.58% | 43.75% | 3.53 pp | -68 | 50 | -1.36 |
| BTC Hourly | nn | NN | 962 | 425 | 537 | 44.18% | 41.67% | 42.50% | 5.82 pp | -112 | 50 | -2.24 |
| BTC Hourly | rf | RandomForest | 962 | 425 | 537 | 44.18% | 42.08% | 42.71% | 5.82 pp | -112 | 50 | -2.24 |
| BTC Hourly | lstm | LSTM | 962 | 410 | 552 | 42.62% | 37.08% | 41.67% | 7.38 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 962 | 397 | 565 | 41.27% | 37.08% | 38.96% | 8.73 pp | -168 | 50 | -3.36 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 785 | 379 | 406 | 48.28% | 46.67% | 47.92% | 1.72 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 785 | 364 | 421 | 46.37% | 40.00% | 46.25% | 3.63 pp | -57 | 46 | -1.24 |
| BTC Daily | nn | NN | 785 | 363 | 422 | 46.24% | 44.58% | 45.00% | 3.76 pp | -59 | 46 | -1.28 |
| BTC Daily | lstm | LSTM | 785 | 329 | 456 | 41.91% | 33.33% | 39.58% | 8.09 pp | -127 | 46 | -2.76 |
| BTC Daily | rf | RandomForest | 785 | 329 | 456 | 41.91% | 38.33% | 41.67% | 8.09 pp | -127 | 46 | -2.76 |
| BTC Daily | xgb | XGBoost | 795 | 310 | 485 | 38.99% | 35.42% | 36.46% | 11.01 pp | -175 | 46 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 47.50% | 47.92% | 1.17 pp | -13 | 52 | -0.25 |
| BTC Market Hours | nn | NN | 557 | 265 | 292 | 47.58% | 51.25% | 49.38% | 2.42 pp | -27 | 52 | -0.52 |
| BTC Market Hours | transformer | Transformer | 557 | 263 | 294 | 47.22% | 47.08% | 47.71% | 2.78 pp | -31 | 52 | -0.60 |
| BTC Market Hours | lstm | LSTM | 557 | 241 | 316 | 43.27% | 41.67% | 43.75% | 6.73 pp | -75 | 52 | -1.44 |
| BTC Market Hours | rf | RandomForest | 557 | 241 | 316 | 43.27% | 45.83% | 43.33% | 6.73 pp | -75 | 52 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 557 | 236 | 321 | 42.37% | 45.00% | 42.71% | 7.63 pp | -85 | 52 | -1.63 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 611 | 287 | 324 | 46.97% | 50.00% | 47.92% | 3.03 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 611 | 286 | 325 | 46.81% | 48.33% | 48.33% | 3.19 pp | -39 | 52 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 611 | 285 | 326 | 46.64% | 49.58% | 47.50% | 3.36 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 611 | 256 | 355 | 41.90% | 44.17% | 41.04% | 8.10 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 611 | 248 | 363 | 40.59% | 40.42% | 40.42% | 9.41 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 611 | 248 | 363 | 40.59% | 42.50% | 40.00% | 9.41 pp | -115 | 52 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 30 | 35 | 46.15% | 46.15% | 46.15% | 3.85 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
