# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T16:35:55.630937+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1311 | 1023 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1187 | 822 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 15:00:00+00:00 | 924 | 584 | 339 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 15:00:00+00:00 | 926 | 638 | 286 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 227 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 584 | 284 | 300 | 48.63% | 47.50% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 584 | 282 | 302 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| BTC Market Hours | transformer | Transformer | 584 | 275 | 309 | 47.09% | 46.67% | 46.67% | 2.91 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 812 | 390 | 422 | 48.03% | 45.83% | 46.88% | 1.97 pp | -32 | 47 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 638 | 300 | 338 | 47.02% | 49.17% | 47.71% | 2.98 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 638 | 300 | 338 | 47.02% | 48.33% | 48.33% | 2.98 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 638 | 299 | 339 | 46.87% | 49.17% | 47.29% | 3.13 pp | -40 | 54 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 989 | 470 | 519 | 47.52% | 50.00% | 46.25% | 2.48 pp | -49 | 51 | -0.96 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 812 | 377 | 435 | 46.43% | 44.58% | 45.21% | 3.57 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 812 | 376 | 436 | 46.31% | 39.17% | 46.25% | 3.69 pp | -60 | 47 | -1.28 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 989 | 458 | 531 | 46.31% | 44.17% | 43.75% | 3.69 pp | -73 | 51 | -1.43 |
| BTC Market Hours | rf | RandomForest | 584 | 251 | 333 | 42.98% | 43.75% | 43.33% | 7.02 pp | -82 | 54 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 584 | 250 | 334 | 42.81% | 44.58% | 43.33% | 7.19 pp | -84 | 54 | -1.56 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| BTC Market Hours | lstm | LSTM | 584 | 249 | 335 | 42.64% | 41.67% | 42.71% | 7.36 pp | -86 | 54 | -1.59 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 638 | 266 | 372 | 41.69% | 42.50% | 41.04% | 8.31 pp | -106 | 54 | -1.96 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 638 | 263 | 375 | 41.22% | 43.75% | 40.62% | 8.78 pp | -112 | 54 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 638 | 262 | 376 | 41.07% | 41.67% | 40.83% | 8.93 pp | -114 | 54 | -2.11 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| BTC Hourly | nn | NN | 989 | 436 | 553 | 44.08% | 42.08% | 42.08% | 5.92 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 989 | 436 | 553 | 44.08% | 41.25% | 42.71% | 5.92 pp | -117 | 51 | -2.29 |
| Consolidated Hourly | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| BTC Daily | lstm | LSTM | 812 | 344 | 468 | 42.36% | 35.83% | 41.04% | 7.64 pp | -124 | 47 | -2.64 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 989 | 420 | 569 | 42.47% | 37.50% | 40.21% | 7.53 pp | -149 | 51 | -2.92 |
| BTC Daily | rf | RandomForest | 812 | 336 | 476 | 41.38% | 36.67% | 41.04% | 8.62 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 989 | 406 | 583 | 41.05% | 35.00% | 38.33% | 8.95 pp | -177 | 51 | -3.47 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 29 | 50 | 36.71% | 36.71% | 36.71% | 13.29 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 822 | 321 | 501 | 39.05% | 35.42% | 35.83% | 10.95 pp | -180 | 47 | -3.83 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 6 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 989 | 470 | 519 | 47.52% | 50.00% | 46.25% | 2.48 pp | -49 | 51 | -0.96 |
| BTC Hourly | transformer | Transformer | 989 | 458 | 531 | 46.31% | 44.17% | 43.75% | 3.69 pp | -73 | 51 | -1.43 |
| BTC Hourly | nn | NN | 989 | 436 | 553 | 44.08% | 42.08% | 42.08% | 5.92 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 989 | 436 | 553 | 44.08% | 41.25% | 42.71% | 5.92 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 989 | 420 | 569 | 42.47% | 37.50% | 40.21% | 7.53 pp | -149 | 51 | -2.92 |
| BTC Hourly | xgb | XGBoost | 989 | 406 | 583 | 41.05% | 35.00% | 38.33% | 8.95 pp | -177 | 51 | -3.47 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 812 | 390 | 422 | 48.03% | 45.83% | 46.88% | 1.97 pp | -32 | 47 | -0.68 |
| BTC Daily | nn | NN | 812 | 377 | 435 | 46.43% | 44.58% | 45.21% | 3.57 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 812 | 376 | 436 | 46.31% | 39.17% | 46.25% | 3.69 pp | -60 | 47 | -1.28 |
| BTC Daily | lstm | LSTM | 812 | 344 | 468 | 42.36% | 35.83% | 41.04% | 7.64 pp | -124 | 47 | -2.64 |
| BTC Daily | rf | RandomForest | 812 | 336 | 476 | 41.38% | 36.67% | 41.04% | 8.62 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 822 | 321 | 501 | 39.05% | 35.42% | 35.83% | 10.95 pp | -180 | 47 | -3.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 584 | 284 | 300 | 48.63% | 47.50% | 47.71% | 1.37 pp | -16 | 54 | -0.30 |
| BTC Market Hours | nn | NN | 584 | 282 | 302 | 48.29% | 52.50% | 50.00% | 1.71 pp | -20 | 54 | -0.37 |
| BTC Market Hours | transformer | Transformer | 584 | 275 | 309 | 47.09% | 46.67% | 46.67% | 2.91 pp | -34 | 54 | -0.63 |
| BTC Market Hours | rf | RandomForest | 584 | 251 | 333 | 42.98% | 43.75% | 43.33% | 7.02 pp | -82 | 54 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 584 | 250 | 334 | 42.81% | 44.58% | 43.33% | 7.19 pp | -84 | 54 | -1.56 |
| BTC Market Hours | lstm | LSTM | 584 | 249 | 335 | 42.64% | 41.67% | 42.71% | 7.36 pp | -86 | 54 | -1.59 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 638 | 300 | 338 | 47.02% | 49.17% | 47.71% | 2.98 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | nn | NN | 638 | 300 | 338 | 47.02% | 48.33% | 48.33% | 2.98 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 638 | 299 | 339 | 46.87% | 49.17% | 47.29% | 3.13 pp | -40 | 54 | -0.74 |
| BTC Market Hours Daily | rf | RandomForest | 638 | 266 | 372 | 41.69% | 42.50% | 41.04% | 8.31 pp | -106 | 54 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 638 | 263 | 375 | 41.22% | 43.75% | 40.62% | 8.78 pp | -112 | 54 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 638 | 262 | 376 | 41.07% | 41.67% | 40.83% | 8.93 pp | -114 | 54 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 29 | 50 | 36.71% | 36.71% | 36.71% | 13.29 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 6 | -3.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
