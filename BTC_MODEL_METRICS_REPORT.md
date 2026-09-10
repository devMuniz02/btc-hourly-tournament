# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T00:58:24.749961+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1316 | 1028 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1192 | 827 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 23:00:00+00:00 | 937 | 589 | 347 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 23:00:00+00:00 | 939 | 643 | 294 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 230 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 589 | 286 | 303 | 48.56% | 48.33% | 47.50% | 1.44 pp | -17 | 55 | -0.31 |
| BTC Market Hours | nn | NN | 589 | 285 | 304 | 48.39% | 52.92% | 50.21% | 1.61 pp | -19 | 55 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| BTC Market Hours | transformer | Transformer | 589 | 277 | 312 | 47.03% | 47.08% | 46.46% | 2.97 pp | -35 | 55 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 817 | 392 | 425 | 47.98% | 45.00% | 46.88% | 2.02 pp | -33 | 47 | -0.70 |
| BTC Market Hours Daily | nn | NN | 643 | 302 | 341 | 46.97% | 48.75% | 48.12% | 3.03 pp | -39 | 55 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 643 | 301 | 342 | 46.81% | 49.17% | 47.08% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 643 | 301 | 342 | 46.81% | 49.17% | 47.71% | 3.19 pp | -41 | 55 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 994 | 472 | 522 | 47.48% | 49.17% | 46.46% | 2.52 pp | -50 | 51 | -0.98 |
| BTC Daily | nn | NN | 817 | 382 | 435 | 46.76% | 45.42% | 45.62% | 3.24 pp | -53 | 47 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| BTC Daily | transformer | Transformer | 817 | 379 | 438 | 46.39% | 40.00% | 46.04% | 3.61 pp | -59 | 47 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 994 | 462 | 532 | 46.48% | 44.58% | 44.38% | 3.52 pp | -70 | 51 | -1.37 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 589 | 252 | 337 | 42.78% | 44.58% | 43.33% | 7.22 pp | -85 | 55 | -1.55 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 589 | 251 | 338 | 42.61% | 42.08% | 42.50% | 7.39 pp | -87 | 55 | -1.58 |
| BTC Market Hours | rf | RandomForest | 589 | 251 | 338 | 42.61% | 43.33% | 42.71% | 7.39 pp | -87 | 55 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 643 | 266 | 377 | 41.37% | 42.08% | 40.83% | 8.63 pp | -111 | 55 | -2.02 |
| BTC Market Hours Daily | xgb | XGBoost | 643 | 264 | 379 | 41.06% | 42.92% | 40.21% | 8.94 pp | -115 | 55 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 643 | 263 | 380 | 40.90% | 42.08% | 40.62% | 9.10 pp | -117 | 55 | -2.13 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Hourly | nn | NN | 994 | 439 | 555 | 44.16% | 42.50% | 42.08% | 5.84 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 994 | 438 | 556 | 44.06% | 41.25% | 42.92% | 5.94 pp | -118 | 51 | -2.31 |
| Consolidated Hourly | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 817 | 346 | 471 | 42.35% | 36.25% | 40.83% | 7.65 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 817 | 339 | 478 | 41.49% | 37.50% | 40.83% | 8.51 pp | -139 | 47 | -2.96 |
| BTC Hourly | lstm | LSTM | 994 | 421 | 573 | 42.35% | 36.67% | 39.79% | 7.65 pp | -152 | 51 | -2.98 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| BTC Hourly | xgb | XGBoost | 994 | 408 | 586 | 41.05% | 34.58% | 38.33% | 8.95 pp | -178 | 51 | -3.49 |
| BTC Daily | xgb | XGBoost | 827 | 325 | 502 | 39.30% | 37.08% | 36.04% | 10.70 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 994 | 472 | 522 | 47.48% | 49.17% | 46.46% | 2.52 pp | -50 | 51 | -0.98 |
| BTC Hourly | transformer | Transformer | 994 | 462 | 532 | 46.48% | 44.58% | 44.38% | 3.52 pp | -70 | 51 | -1.37 |
| BTC Hourly | nn | NN | 994 | 439 | 555 | 44.16% | 42.50% | 42.08% | 5.84 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 994 | 438 | 556 | 44.06% | 41.25% | 42.92% | 5.94 pp | -118 | 51 | -2.31 |
| BTC Hourly | lstm | LSTM | 994 | 421 | 573 | 42.35% | 36.67% | 39.79% | 7.65 pp | -152 | 51 | -2.98 |
| BTC Hourly | xgb | XGBoost | 994 | 408 | 586 | 41.05% | 34.58% | 38.33% | 8.95 pp | -178 | 51 | -3.49 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 817 | 392 | 425 | 47.98% | 45.00% | 46.88% | 2.02 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 817 | 382 | 435 | 46.76% | 45.42% | 45.62% | 3.24 pp | -53 | 47 | -1.13 |
| BTC Daily | transformer | Transformer | 817 | 379 | 438 | 46.39% | 40.00% | 46.04% | 3.61 pp | -59 | 47 | -1.26 |
| BTC Daily | lstm | LSTM | 817 | 346 | 471 | 42.35% | 36.25% | 40.83% | 7.65 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 817 | 339 | 478 | 41.49% | 37.50% | 40.83% | 8.51 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 827 | 325 | 502 | 39.30% | 37.08% | 36.04% | 10.70 pp | -177 | 47 | -3.77 |

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
| Consolidated Hourly | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
