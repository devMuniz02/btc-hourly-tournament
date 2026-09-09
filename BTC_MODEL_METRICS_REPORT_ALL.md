# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T01:19:54.474842+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1301 | 1013 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1177 | 812 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 910 | 574 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 911 | 627 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 215 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 17:00:00+00:00 | 215 | 72 | 143 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 574 | 280 | 294 | 48.78% | 47.92% | 48.12% | 1.22 pp | -14 | 54 | -0.26 |
| BTC Market Hours | nn | NN | 574 | 274 | 300 | 47.74% | 52.08% | 49.38% | 2.26 pp | -26 | 54 | -0.48 |
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 574 | 270 | 304 | 47.04% | 46.67% | 46.88% | 2.96 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| BTC Market Hours Daily | nn | NN | 627 | 293 | 334 | 46.73% | 47.92% | 48.12% | 3.27 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 627 | 292 | 335 | 46.57% | 48.75% | 47.08% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 627 | 292 | 335 | 46.57% | 48.75% | 47.29% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 802 | 373 | 429 | 46.51% | 45.00% | 44.79% | 3.49 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 574 | 249 | 325 | 43.38% | 43.33% | 43.75% | 6.62 pp | -76 | 54 | -1.41 |
| BTC Market Hours | rf | RandomForest | 574 | 248 | 326 | 43.21% | 45.42% | 43.75% | 6.79 pp | -78 | 54 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 574 | 248 | 326 | 43.21% | 47.08% | 43.54% | 6.79 pp | -78 | 54 | -1.44 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 627 | 261 | 366 | 41.63% | 42.92% | 40.62% | 8.37 pp | -105 | 53 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 627 | 259 | 368 | 41.31% | 44.17% | 40.83% | 8.69 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 627 | 255 | 372 | 40.67% | 41.25% | 40.00% | 9.33 pp | -117 | 53 | -2.21 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 802 | 337 | 465 | 42.02% | 34.58% | 40.00% | 7.98 pp | -128 | 46 | -2.78 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 802 | 334 | 468 | 41.65% | 37.08% | 41.04% | 8.35 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 812 | 316 | 496 | 38.92% | 35.00% | 35.62% | 11.08 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| BTC Daily | nn | NN | 802 | 373 | 429 | 46.51% | 45.00% | 44.79% | 3.49 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 802 | 337 | 465 | 42.02% | 34.58% | 40.00% | 7.98 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 802 | 334 | 468 | 41.65% | 37.08% | 41.04% | 8.35 pp | -134 | 46 | -2.91 |
| BTC Daily | xgb | XGBoost | 812 | 316 | 496 | 38.92% | 35.00% | 35.62% | 11.08 pp | -180 | 46 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 574 | 280 | 294 | 48.78% | 47.92% | 48.12% | 1.22 pp | -14 | 54 | -0.26 |
| BTC Market Hours | nn | NN | 574 | 274 | 300 | 47.74% | 52.08% | 49.38% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 574 | 270 | 304 | 47.04% | 46.67% | 46.88% | 2.96 pp | -34 | 54 | -0.63 |
| BTC Market Hours | lstm | LSTM | 574 | 249 | 325 | 43.38% | 43.33% | 43.75% | 6.62 pp | -76 | 54 | -1.41 |
| BTC Market Hours | rf | RandomForest | 574 | 248 | 326 | 43.21% | 45.42% | 43.75% | 6.79 pp | -78 | 54 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 574 | 248 | 326 | 43.21% | 47.08% | 43.54% | 6.79 pp | -78 | 54 | -1.44 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 627 | 293 | 334 | 46.73% | 47.92% | 48.12% | 3.27 pp | -41 | 53 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 627 | 292 | 335 | 46.57% | 48.75% | 47.08% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 627 | 292 | 335 | 46.57% | 48.75% | 47.29% | 3.43 pp | -43 | 53 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 627 | 261 | 366 | 41.63% | 42.92% | 40.62% | 8.37 pp | -105 | 53 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 627 | 259 | 368 | 41.31% | 44.17% | 40.83% | 8.69 pp | -109 | 53 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 627 | 255 | 372 | 40.67% | 41.25% | 40.00% | 9.33 pp | -117 | 53 | -2.21 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 104 | 111 | 48.37% | 48.37% | 48.37% | 1.63 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 100 | 115 | 46.51% | 46.51% | 46.51% | 3.49 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 95 | 120 | 44.19% | 44.19% | 44.19% | 5.81 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 94 | 121 | 43.72% | 43.72% | 43.72% | 6.28 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 14 | -2.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
