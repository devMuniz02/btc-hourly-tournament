# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T05:39:34.754006+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1304 | 1016 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1179 | 814 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 912 | 576 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 914 | 630 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 219 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 19:00:00+00:00 | 219 | 74 | 145 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 576 | 280 | 296 | 48.61% | 47.08% | 47.71% | 1.39 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 576 | 275 | 301 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 54 | -0.48 |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 576 | 272 | 304 | 47.22% | 47.50% | 47.08% | 2.78 pp | -32 | 54 | -0.59 |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Market Hours Daily | transformer | Transformer | 630 | 295 | 335 | 46.83% | 49.17% | 47.50% | 3.17 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | nn | NN | 630 | 294 | 336 | 46.67% | 47.08% | 47.92% | 3.33 pp | -42 | 54 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 630 | 293 | 337 | 46.51% | 47.92% | 46.67% | 3.49 pp | -44 | 54 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 982 | 469 | 513 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 372 | 432 | 46.27% | 39.17% | 45.83% | 3.73 pp | -60 | 46 | -1.30 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 982 | 455 | 527 | 46.33% | 43.75% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 576 | 249 | 327 | 43.23% | 42.50% | 43.33% | 6.77 pp | -78 | 54 | -1.44 |
| BTC Market Hours | rf | RandomForest | 576 | 248 | 328 | 43.06% | 45.00% | 43.33% | 6.94 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 576 | 248 | 328 | 43.06% | 46.25% | 43.12% | 6.94 pp | -80 | 54 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 630 | 263 | 367 | 41.75% | 43.75% | 40.62% | 8.25 pp | -104 | 54 | -1.93 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 630 | 261 | 369 | 41.43% | 45.00% | 40.83% | 8.57 pp | -108 | 54 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 630 | 258 | 372 | 40.95% | 42.08% | 40.21% | 9.05 pp | -114 | 54 | -2.11 |
| BTC Hourly | rf | RandomForest | 982 | 434 | 548 | 44.20% | 41.67% | 42.71% | 5.80 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 982 | 433 | 549 | 44.09% | 41.67% | 42.08% | 5.91 pp | -116 | 51 | -2.27 |
| BTC Daily | lstm | LSTM | 804 | 340 | 464 | 42.29% | 35.42% | 40.62% | 7.71 pp | -124 | 46 | -2.70 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |
| BTC Hourly | lstm | LSTM | 982 | 418 | 564 | 42.57% | 37.92% | 40.83% | 7.43 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 804 | 333 | 471 | 41.42% | 36.67% | 40.83% | 8.58 pp | -138 | 46 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 982 | 405 | 577 | 41.24% | 35.83% | 38.96% | 8.76 pp | -172 | 51 | -3.37 |
| BTC Daily | xgb | XGBoost | 814 | 316 | 498 | 38.82% | 35.00% | 35.42% | 11.18 pp | -182 | 46 | -3.96 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 982 | 469 | 513 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| BTC Hourly | transformer | Transformer | 982 | 455 | 527 | 46.33% | 43.75% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Hourly | rf | RandomForest | 982 | 434 | 548 | 44.20% | 41.67% | 42.71% | 5.80 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 982 | 433 | 549 | 44.09% | 41.67% | 42.08% | 5.91 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 982 | 418 | 564 | 42.57% | 37.92% | 40.83% | 7.43 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 982 | 405 | 577 | 41.24% | 35.83% | 38.96% | 8.76 pp | -172 | 51 | -3.37 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 372 | 432 | 46.27% | 39.17% | 45.83% | 3.73 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 804 | 340 | 464 | 42.29% | 35.42% | 40.62% | 7.71 pp | -124 | 46 | -2.70 |
| BTC Daily | rf | RandomForest | 804 | 333 | 471 | 41.42% | 36.67% | 40.83% | 8.58 pp | -138 | 46 | -3.00 |
| BTC Daily | xgb | XGBoost | 814 | 316 | 498 | 38.82% | 35.00% | 35.42% | 11.18 pp | -182 | 46 | -3.96 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 576 | 280 | 296 | 48.61% | 47.08% | 47.71% | 1.39 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 576 | 275 | 301 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 576 | 272 | 304 | 47.22% | 47.50% | 47.08% | 2.78 pp | -32 | 54 | -0.59 |
| BTC Market Hours | lstm | LSTM | 576 | 249 | 327 | 43.23% | 42.50% | 43.33% | 6.77 pp | -78 | 54 | -1.44 |
| BTC Market Hours | rf | RandomForest | 576 | 248 | 328 | 43.06% | 45.00% | 43.33% | 6.94 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 576 | 248 | 328 | 43.06% | 46.25% | 43.12% | 6.94 pp | -80 | 54 | -1.48 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 630 | 295 | 335 | 46.83% | 49.17% | 47.50% | 3.17 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | nn | NN | 630 | 294 | 336 | 46.67% | 47.08% | 47.92% | 3.33 pp | -42 | 54 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 630 | 293 | 337 | 46.51% | 47.92% | 46.67% | 3.49 pp | -44 | 54 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 630 | 263 | 367 | 41.75% | 43.75% | 40.62% | 8.25 pp | -104 | 54 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 630 | 261 | 369 | 41.43% | 45.00% | 40.83% | 8.57 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 630 | 258 | 372 | 40.95% | 42.08% | 40.21% | 9.05 pp | -114 | 54 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 106 | 113 | 48.40% | 48.40% | 48.40% | 1.60 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 101 | 118 | 46.12% | 46.12% | 46.12% | 3.88 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 96 | 123 | 43.84% | 43.84% | 43.84% | 6.16 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 95 | 124 | 43.38% | 43.38% | 43.38% | 6.62 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 90 | 129 | 41.10% | 41.10% | 41.10% | 8.90 pp | -39 | 14 | -2.79 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
