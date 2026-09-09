# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T18:29:42.013261+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1312 | 1024 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1188 | 823 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 17:00:00+00:00 | 927 | 585 | 341 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 17:00:00+00:00 | 929 | 639 | 288 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 227 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 585 | 285 | 300 | 48.72% | 47.92% | 47.71% | 1.28 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 585 | 283 | 302 | 48.38% | 52.92% | 50.21% | 1.62 pp | -19 | 54 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| BTC Market Hours | transformer | Transformer | 585 | 276 | 309 | 47.18% | 47.08% | 46.67% | 2.82 pp | -33 | 54 | -0.61 |
| BTC Daily | mlp_sklearn | MLPClassifier | 813 | 391 | 422 | 48.09% | 45.83% | 47.08% | 1.91 pp | -31 | 47 | -0.66 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 639 | 300 | 339 | 46.95% | 49.17% | 47.50% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 639 | 300 | 339 | 46.95% | 48.33% | 48.33% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 639 | 299 | 340 | 46.79% | 48.75% | 47.29% | 3.21 pp | -41 | 54 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 990 | 470 | 520 | 47.47% | 49.58% | 46.25% | 2.53 pp | -50 | 51 | -0.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 813 | 378 | 435 | 46.49% | 44.58% | 45.21% | 3.51 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 813 | 377 | 436 | 46.37% | 39.58% | 46.25% | 3.63 pp | -59 | 47 | -1.26 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 990 | 459 | 531 | 46.36% | 44.17% | 43.96% | 3.64 pp | -72 | 51 | -1.41 |
| BTC Market Hours | rf | RandomForest | 585 | 251 | 334 | 42.91% | 43.33% | 43.12% | 7.09 pp | -83 | 54 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 585 | 251 | 334 | 42.91% | 44.58% | 43.33% | 7.09 pp | -83 | 54 | -1.54 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| BTC Market Hours | lstm | LSTM | 585 | 250 | 335 | 42.74% | 42.08% | 42.71% | 7.26 pp | -85 | 54 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 639 | 266 | 373 | 41.63% | 42.50% | 41.04% | 8.37 pp | -107 | 54 | -1.98 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 639 | 263 | 376 | 41.16% | 43.33% | 40.42% | 8.84 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 639 | 262 | 377 | 41.00% | 41.67% | 40.83% | 9.00 pp | -115 | 54 | -2.13 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| BTC Hourly | nn | NN | 990 | 436 | 554 | 44.04% | 41.67% | 42.08% | 5.96 pp | -118 | 51 | -2.31 |
| BTC Hourly | rf | RandomForest | 990 | 436 | 554 | 44.04% | 41.25% | 42.71% | 5.96 pp | -118 | 51 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Hourly | transformer | Transformer | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| BTC Daily | lstm | LSTM | 813 | 344 | 469 | 42.31% | 35.83% | 40.83% | 7.69 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 990 | 420 | 570 | 42.42% | 37.08% | 40.21% | 7.58 pp | -150 | 51 | -2.94 |
| BTC Daily | rf | RandomForest | 813 | 337 | 476 | 41.45% | 37.08% | 41.04% | 8.55 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 990 | 406 | 584 | 41.01% | 35.00% | 38.33% | 8.99 pp | -178 | 51 | -3.49 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 29 | 50 | 36.71% | 36.71% | 36.71% | 13.29 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 823 | 322 | 501 | 39.13% | 35.83% | 35.83% | 10.87 pp | -179 | 47 | -3.81 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 6 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 990 | 470 | 520 | 47.47% | 49.58% | 46.25% | 2.53 pp | -50 | 51 | -0.98 |
| BTC Hourly | transformer | Transformer | 990 | 459 | 531 | 46.36% | 44.17% | 43.96% | 3.64 pp | -72 | 51 | -1.41 |
| BTC Hourly | nn | NN | 990 | 436 | 554 | 44.04% | 41.67% | 42.08% | 5.96 pp | -118 | 51 | -2.31 |
| BTC Hourly | rf | RandomForest | 990 | 436 | 554 | 44.04% | 41.25% | 42.71% | 5.96 pp | -118 | 51 | -2.31 |
| BTC Hourly | lstm | LSTM | 990 | 420 | 570 | 42.42% | 37.08% | 40.21% | 7.58 pp | -150 | 51 | -2.94 |
| BTC Hourly | xgb | XGBoost | 990 | 406 | 584 | 41.01% | 35.00% | 38.33% | 8.99 pp | -178 | 51 | -3.49 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 813 | 391 | 422 | 48.09% | 45.83% | 47.08% | 1.91 pp | -31 | 47 | -0.66 |
| BTC Daily | nn | NN | 813 | 378 | 435 | 46.49% | 44.58% | 45.21% | 3.51 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 813 | 377 | 436 | 46.37% | 39.58% | 46.25% | 3.63 pp | -59 | 47 | -1.26 |
| BTC Daily | lstm | LSTM | 813 | 344 | 469 | 42.31% | 35.83% | 40.83% | 7.69 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 813 | 337 | 476 | 41.45% | 37.08% | 41.04% | 8.55 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 823 | 322 | 501 | 39.13% | 35.83% | 35.83% | 10.87 pp | -179 | 47 | -3.81 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 585 | 285 | 300 | 48.72% | 47.92% | 47.71% | 1.28 pp | -15 | 54 | -0.28 |
| BTC Market Hours | nn | NN | 585 | 283 | 302 | 48.38% | 52.92% | 50.21% | 1.62 pp | -19 | 54 | -0.35 |
| BTC Market Hours | transformer | Transformer | 585 | 276 | 309 | 47.18% | 47.08% | 46.67% | 2.82 pp | -33 | 54 | -0.61 |
| BTC Market Hours | rf | RandomForest | 585 | 251 | 334 | 42.91% | 43.33% | 43.12% | 7.09 pp | -83 | 54 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 585 | 251 | 334 | 42.91% | 44.58% | 43.33% | 7.09 pp | -83 | 54 | -1.54 |
| BTC Market Hours | lstm | LSTM | 585 | 250 | 335 | 42.74% | 42.08% | 42.71% | 7.26 pp | -85 | 54 | -1.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 639 | 300 | 339 | 46.95% | 49.17% | 47.50% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 639 | 300 | 339 | 46.95% | 48.33% | 48.33% | 3.05 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 639 | 299 | 340 | 46.79% | 48.75% | 47.29% | 3.21 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 639 | 266 | 373 | 41.63% | 42.50% | 41.04% | 8.37 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 639 | 263 | 376 | 41.16% | 43.33% | 40.42% | 8.84 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 639 | 262 | 377 | 41.00% | 41.67% | 40.83% | 9.00 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Hourly | transformer | Transformer | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
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
