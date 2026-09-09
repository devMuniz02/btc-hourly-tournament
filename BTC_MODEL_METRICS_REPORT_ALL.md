# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T05:12:04.626988+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1303 | 1015 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1179 | 814 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 912 | 576 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 914 | 630 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 219 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T19:00:00+00:00 | 220 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 576 | 280 | 296 | 48.61% | 47.08% | 47.71% | 1.39 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 576 | 275 | 301 | 47.74% | 51.67% | 49.17% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 576 | 272 | 304 | 47.22% | 47.50% | 47.08% | 2.78 pp | -32 | 54 | -0.59 |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Market Hours Daily | transformer | Transformer | 630 | 295 | 335 | 46.83% | 49.17% | 47.50% | 3.17 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | nn | NN | 630 | 294 | 336 | 46.67% | 47.08% | 47.92% | 3.33 pp | -42 | 54 | -0.78 |
| Consolidated Hourly | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 630 | 293 | 337 | 46.51% | 47.92% | 46.67% | 3.49 pp | -44 | 54 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 981 | 469 | 512 | 47.81% | 51.25% | 46.88% | 2.19 pp | -43 | 51 | -0.84 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 373 | 431 | 46.39% | 39.58% | 46.04% | 3.61 pp | -58 | 46 | -1.26 |
| Consolidated Market Hours | xgb | XGBoost | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 981 | 454 | 527 | 46.28% | 43.33% | 43.33% | 3.72 pp | -73 | 51 | -1.43 |
| BTC Market Hours | lstm | LSTM | 576 | 249 | 327 | 43.23% | 42.50% | 43.33% | 6.77 pp | -78 | 54 | -1.44 |
| BTC Market Hours | rf | RandomForest | 576 | 248 | 328 | 43.06% | 45.00% | 43.33% | 6.94 pp | -80 | 54 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 576 | 248 | 328 | 43.06% | 46.25% | 43.12% | 6.94 pp | -80 | 54 | -1.48 |
| Consolidated Hourly | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 630 | 263 | 367 | 41.75% | 43.75% | 40.62% | 8.25 pp | -104 | 54 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 630 | 261 | 369 | 41.43% | 45.00% | 40.83% | 8.57 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 630 | 258 | 372 | 40.95% | 42.08% | 40.21% | 9.05 pp | -114 | 54 | -2.11 |
| Consolidated Hourly | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 981 | 434 | 547 | 44.24% | 42.08% | 42.71% | 5.76 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 981 | 433 | 548 | 44.14% | 41.67% | 42.29% | 5.86 pp | -115 | 51 | -2.25 |
| BTC Daily | lstm | LSTM | 804 | 339 | 465 | 42.16% | 35.42% | 40.42% | 7.84 pp | -126 | 46 | -2.74 |
| Consolidated Market Hours Daily | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 981 | 417 | 564 | 42.51% | 37.50% | 40.83% | 7.49 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 804 | 334 | 470 | 41.54% | 37.08% | 41.04% | 8.46 pp | -136 | 46 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 981 | 405 | 576 | 41.28% | 36.25% | 38.96% | 8.72 pp | -171 | 51 | -3.35 |
| BTC Daily | xgb | XGBoost | 814 | 316 | 498 | 38.82% | 35.00% | 35.42% | 11.18 pp | -182 | 46 | -3.96 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 981 | 469 | 512 | 47.81% | 51.25% | 46.88% | 2.19 pp | -43 | 51 | -0.84 |
| BTC Hourly | transformer | Transformer | 981 | 454 | 527 | 46.28% | 43.33% | 43.33% | 3.72 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 981 | 434 | 547 | 44.24% | 42.08% | 42.71% | 5.76 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 981 | 433 | 548 | 44.14% | 41.67% | 42.29% | 5.86 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 981 | 417 | 564 | 42.51% | 37.50% | 40.83% | 7.49 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 981 | 405 | 576 | 41.28% | 36.25% | 38.96% | 8.72 pp | -171 | 51 | -3.35 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 804 | 388 | 416 | 48.26% | 46.67% | 47.29% | 1.74 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 804 | 373 | 431 | 46.39% | 44.17% | 44.79% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 804 | 373 | 431 | 46.39% | 39.58% | 46.04% | 3.61 pp | -58 | 46 | -1.26 |
| BTC Daily | lstm | LSTM | 804 | 339 | 465 | 42.16% | 35.42% | 40.42% | 7.84 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 804 | 334 | 470 | 41.54% | 37.08% | 41.04% | 8.46 pp | -136 | 46 | -2.96 |
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
| Consolidated Hourly | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 219 | 104 | 115 | 47.49% | 47.49% | 47.49% | 2.51 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 219 | 102 | 117 | 46.58% | 46.58% | 46.58% | 3.42 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 219 | 99 | 120 | 45.21% | 45.21% | 45.21% | 4.79 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 219 | 94 | 125 | 42.92% | 42.92% | 42.92% | 7.08 pp | -31 | 14 | -2.21 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
