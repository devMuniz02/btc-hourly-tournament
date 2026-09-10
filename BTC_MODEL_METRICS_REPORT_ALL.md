# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T04:04:26.171593+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1318 | 1030 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1194 | 829 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 940 | 591 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 942 | 645 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 231 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 231 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 231 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 232 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 591 | 286 | 305 | 48.39% | 47.50% | 47.08% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 591 | 285 | 306 | 48.22% | 52.08% | 49.79% | 1.78 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 591 | 277 | 314 | 46.87% | 46.67% | 46.25% | 3.13 pp | -37 | 55 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 819 | 393 | 426 | 47.99% | 45.42% | 47.08% | 2.01 pp | -33 | 47 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 645 | 303 | 342 | 46.98% | 49.17% | 47.50% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | nn | NN | 645 | 303 | 342 | 46.98% | 48.75% | 48.12% | 3.02 pp | -39 | 55 | -0.71 |
| Consolidated Hourly | rf | RandomForest | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 645 | 301 | 344 | 46.67% | 48.75% | 47.71% | 3.33 pp | -43 | 55 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 996 | 472 | 524 | 47.39% | 48.75% | 46.25% | 2.61 pp | -52 | 51 | -1.02 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 819 | 382 | 437 | 46.64% | 45.42% | 45.62% | 3.36 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 819 | 380 | 439 | 46.40% | 39.58% | 45.83% | 3.60 pp | -59 | 47 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 996 | 463 | 533 | 46.49% | 45.00% | 44.58% | 3.51 pp | -70 | 51 | -1.37 |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 591 | 253 | 338 | 42.81% | 44.58% | 43.33% | 7.19 pp | -85 | 55 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 591 | 252 | 339 | 42.64% | 42.92% | 42.50% | 7.36 pp | -87 | 55 | -1.58 |
| BTC Market Hours | lstm | LSTM | 591 | 251 | 340 | 42.47% | 42.08% | 42.29% | 7.53 pp | -89 | 55 | -1.62 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | xgb | XGBoost | 231 | 102 | 129 | 44.16% | 44.16% | 44.16% | 5.84 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 102 | 129 | 44.16% | 44.16% | 44.16% | 5.84 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 645 | 266 | 379 | 41.24% | 41.67% | 40.62% | 8.76 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 645 | 265 | 380 | 41.09% | 42.92% | 40.21% | 8.91 pp | -115 | 55 | -2.09 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 645 | 263 | 382 | 40.78% | 41.67% | 40.21% | 9.22 pp | -119 | 55 | -2.16 |
| Consolidated Hourly | transformer | Transformer | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 996 | 440 | 556 | 44.18% | 42.50% | 42.08% | 5.82 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 996 | 439 | 557 | 44.08% | 41.25% | 43.12% | 5.92 pp | -118 | 51 | -2.31 |
| Consolidated Hourly | nn | NN | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 819 | 347 | 472 | 42.37% | 36.67% | 41.04% | 7.63 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 819 | 340 | 479 | 41.51% | 37.92% | 41.04% | 8.49 pp | -139 | 47 | -2.96 |
| BTC Hourly | lstm | LSTM | 996 | 422 | 574 | 42.37% | 37.08% | 40.00% | 7.63 pp | -152 | 51 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 996 | 410 | 586 | 41.16% | 35.00% | 38.54% | 8.84 pp | -176 | 51 | -3.45 |
| BTC Daily | xgb | XGBoost | 829 | 326 | 503 | 39.32% | 37.50% | 36.25% | 10.68 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 996 | 472 | 524 | 47.39% | 48.75% | 46.25% | 2.61 pp | -52 | 51 | -1.02 |
| BTC Hourly | transformer | Transformer | 996 | 463 | 533 | 46.49% | 45.00% | 44.58% | 3.51 pp | -70 | 51 | -1.37 |
| BTC Hourly | nn | NN | 996 | 440 | 556 | 44.18% | 42.50% | 42.08% | 5.82 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 996 | 439 | 557 | 44.08% | 41.25% | 43.12% | 5.92 pp | -118 | 51 | -2.31 |
| BTC Hourly | lstm | LSTM | 996 | 422 | 574 | 42.37% | 37.08% | 40.00% | 7.63 pp | -152 | 51 | -2.98 |
| BTC Hourly | xgb | XGBoost | 996 | 410 | 586 | 41.16% | 35.00% | 38.54% | 8.84 pp | -176 | 51 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 819 | 393 | 426 | 47.99% | 45.42% | 47.08% | 2.01 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 819 | 382 | 437 | 46.64% | 45.42% | 45.62% | 3.36 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 819 | 380 | 439 | 46.40% | 39.58% | 45.83% | 3.60 pp | -59 | 47 | -1.26 |
| BTC Daily | lstm | LSTM | 819 | 347 | 472 | 42.37% | 36.67% | 41.04% | 7.63 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 819 | 340 | 479 | 41.51% | 37.92% | 41.04% | 8.49 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 829 | 326 | 503 | 39.32% | 37.50% | 36.25% | 10.68 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 591 | 286 | 305 | 48.39% | 47.50% | 47.08% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | nn | NN | 591 | 285 | 306 | 48.22% | 52.08% | 49.79% | 1.78 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 591 | 277 | 314 | 46.87% | 46.67% | 46.25% | 3.13 pp | -37 | 55 | -0.67 |
| BTC Market Hours | xgb | XGBoost | 591 | 253 | 338 | 42.81% | 44.58% | 43.33% | 7.19 pp | -85 | 55 | -1.55 |
| BTC Market Hours | rf | RandomForest | 591 | 252 | 339 | 42.64% | 42.92% | 42.50% | 7.36 pp | -87 | 55 | -1.58 |
| BTC Market Hours | lstm | LSTM | 591 | 251 | 340 | 42.47% | 42.08% | 42.29% | 7.53 pp | -89 | 55 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 645 | 303 | 342 | 46.98% | 49.17% | 47.50% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | nn | NN | 645 | 303 | 342 | 46.98% | 48.75% | 48.12% | 3.02 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 645 | 301 | 344 | 46.67% | 48.75% | 47.71% | 3.33 pp | -43 | 55 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 645 | 266 | 379 | 41.24% | 41.67% | 40.62% | 8.76 pp | -113 | 55 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 645 | 265 | 380 | 41.09% | 42.92% | 40.21% | 8.91 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 645 | 263 | 382 | 40.78% | 41.67% | 40.21% | 9.22 pp | -119 | 55 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | xgb | XGBoost | 231 | 102 | 129 | 44.16% | 44.16% | 44.16% | 5.84 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 102 | 129 | 44.16% | 44.16% | 44.16% | 5.84 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
