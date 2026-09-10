# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T01:38:16.966108+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1317 | 1029 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1192 | 827 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 938 | 589 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 940 | 643 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 589 | 286 | 303 | 48.56% | 48.33% | 47.50% | 1.44 pp | -17 | 55 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| BTC Market Hours | nn | NN | 589 | 285 | 304 | 48.39% | 52.92% | 50.21% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | transformer | Transformer | 589 | 277 | 312 | 47.03% | 47.08% | 46.46% | 2.97 pp | -35 | 55 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 817 | 392 | 425 | 47.98% | 45.42% | 46.88% | 2.02 pp | -33 | 47 | -0.70 |
| BTC Market Hours Daily | nn | NN | 643 | 302 | 341 | 46.97% | 48.75% | 48.12% | 3.03 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 643 | 301 | 342 | 46.81% | 49.17% | 47.08% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 643 | 301 | 342 | 46.81% | 49.17% | 47.71% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 995 | 472 | 523 | 47.44% | 48.75% | 46.25% | 2.56 pp | -51 | 51 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 817 | 381 | 436 | 46.63% | 45.42% | 45.42% | 3.37 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 817 | 378 | 439 | 46.27% | 39.58% | 45.83% | 3.73 pp | -61 | 47 | -1.30 |
| BTC Hourly | transformer | Transformer | 995 | 463 | 532 | 46.53% | 45.00% | 44.58% | 3.47 pp | -69 | 51 | -1.35 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 589 | 252 | 337 | 42.78% | 44.58% | 43.33% | 7.22 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 589 | 251 | 338 | 42.61% | 42.08% | 42.50% | 7.39 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 589 | 251 | 338 | 42.61% | 43.33% | 42.71% | 7.39 pp | -87 | 55 | -1.58 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 643 | 266 | 377 | 41.37% | 42.08% | 40.83% | 8.63 pp | -111 | 55 | -2.02 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 643 | 264 | 379 | 41.06% | 42.92% | 40.21% | 8.94 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 643 | 263 | 380 | 40.90% | 42.08% | 40.62% | 9.10 pp | -117 | 55 | -2.13 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 995 | 439 | 556 | 44.12% | 42.08% | 41.88% | 5.88 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 995 | 438 | 557 | 44.02% | 40.83% | 42.92% | 5.98 pp | -119 | 51 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 817 | 347 | 470 | 42.47% | 36.67% | 41.04% | 7.53 pp | -123 | 47 | -2.62 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 817 | 339 | 478 | 41.49% | 37.92% | 40.83% | 8.51 pp | -139 | 47 | -2.96 |
| BTC Hourly | lstm | LSTM | 995 | 421 | 574 | 42.31% | 36.67% | 39.79% | 7.69 pp | -153 | 51 | -3.00 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| BTC Hourly | xgb | XGBoost | 995 | 409 | 586 | 41.11% | 34.58% | 38.33% | 8.89 pp | -177 | 51 | -3.47 |
| BTC Daily | xgb | XGBoost | 827 | 325 | 502 | 39.30% | 37.50% | 36.04% | 10.70 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 995 | 472 | 523 | 47.44% | 48.75% | 46.25% | 2.56 pp | -51 | 51 | -1.00 |
| BTC Hourly | transformer | Transformer | 995 | 463 | 532 | 46.53% | 45.00% | 44.58% | 3.47 pp | -69 | 51 | -1.35 |
| BTC Hourly | nn | NN | 995 | 439 | 556 | 44.12% | 42.08% | 41.88% | 5.88 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 995 | 438 | 557 | 44.02% | 40.83% | 42.92% | 5.98 pp | -119 | 51 | -2.33 |
| BTC Hourly | lstm | LSTM | 995 | 421 | 574 | 42.31% | 36.67% | 39.79% | 7.69 pp | -153 | 51 | -3.00 |
| BTC Hourly | xgb | XGBoost | 995 | 409 | 586 | 41.11% | 34.58% | 38.33% | 8.89 pp | -177 | 51 | -3.47 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 817 | 392 | 425 | 47.98% | 45.42% | 46.88% | 2.02 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 817 | 381 | 436 | 46.63% | 45.42% | 45.42% | 3.37 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 817 | 378 | 439 | 46.27% | 39.58% | 45.83% | 3.73 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 817 | 347 | 470 | 42.47% | 36.67% | 41.04% | 7.53 pp | -123 | 47 | -2.62 |
| BTC Daily | rf | RandomForest | 817 | 339 | 478 | 41.49% | 37.92% | 40.83% | 8.51 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 827 | 325 | 502 | 39.30% | 37.50% | 36.04% | 10.70 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 589 | 286 | 303 | 48.56% | 48.33% | 47.50% | 1.44 pp | -17 | 55 | -0.31 |
| BTC Market Hours | nn | NN | 589 | 285 | 304 | 48.39% | 52.92% | 50.21% | 1.61 pp | -19 | 55 | -0.35 |
| BTC Market Hours | transformer | Transformer | 589 | 277 | 312 | 47.03% | 47.08% | 46.46% | 2.97 pp | -35 | 55 | -0.64 |
| BTC Market Hours | xgb | XGBoost | 589 | 252 | 337 | 42.78% | 44.58% | 43.33% | 7.22 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 589 | 251 | 338 | 42.61% | 42.08% | 42.50% | 7.39 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 589 | 251 | 338 | 42.61% | 43.33% | 42.71% | 7.39 pp | -87 | 55 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 643 | 302 | 341 | 46.97% | 48.75% | 48.12% | 3.03 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 643 | 301 | 342 | 46.81% | 49.17% | 47.08% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 643 | 301 | 342 | 46.81% | 49.17% | 47.71% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 643 | 266 | 377 | 41.37% | 42.08% | 40.83% | 8.63 pp | -111 | 55 | -2.02 |
| BTC Market Hours Daily | xgb | XGBoost | 643 | 264 | 379 | 41.06% | 42.92% | 40.21% | 8.94 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 643 | 263 | 380 | 40.90% | 42.08% | 40.62% | 9.10 pp | -117 | 55 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
