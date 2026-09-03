# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T06:56:49.634498+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1207 | 919 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1083 | 718 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 738 | 480 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 740 | 534 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 131 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 708 | 346 | 362 | 48.87% | 47.50% | 48.75% | 1.13 pp | -16 | 42 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 480 | 231 | 249 | 48.12% | 43.75% | 48.12% | 1.87 pp | -18 | 46 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 480 | 226 | 254 | 47.08% | 48.33% | 47.08% | 2.92 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 480 | 224 | 256 | 46.67% | 41.67% | 46.67% | 3.33 pp | -32 | 46 | -0.70 |
| BTC Daily | transformer | Transformer | 708 | 339 | 369 | 47.88% | 47.08% | 50.00% | 2.12 pp | -30 | 42 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 534 | 248 | 286 | 46.44% | 49.58% | 47.50% | 3.56 pp | -38 | 46 | -0.83 |
| BTC Market Hours Daily | nn | NN | 534 | 245 | 289 | 45.88% | 43.75% | 46.67% | 4.12 pp | -44 | 46 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 885 | 420 | 465 | 47.46% | 49.17% | 47.92% | 2.54 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 885 | 420 | 465 | 47.46% | 48.75% | 47.92% | 2.54 pp | -45 | 47 | -0.96 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 534 | 243 | 291 | 45.51% | 47.08% | 46.46% | 4.49 pp | -48 | 46 | -1.04 |
| BTC Daily | nn | NN | 708 | 328 | 380 | 46.33% | 43.33% | 47.92% | 3.67 pp | -52 | 42 | -1.24 |
| BTC Market Hours | lstm | LSTM | 480 | 207 | 273 | 43.12% | 41.67% | 43.12% | 6.87 pp | -66 | 46 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| BTC Market Hours | rf | RandomForest | 480 | 205 | 275 | 42.71% | 42.08% | 42.71% | 7.29 pp | -70 | 46 | -1.52 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 480 | 195 | 285 | 40.62% | 39.58% | 40.62% | 9.38 pp | -90 | 46 | -1.96 |
| BTC Hourly | nn | NN | 885 | 396 | 489 | 44.75% | 45.42% | 43.33% | 5.25 pp | -93 | 47 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 885 | 395 | 490 | 44.63% | 44.58% | 44.38% | 5.37 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 534 | 219 | 315 | 41.01% | 40.83% | 41.25% | 8.99 pp | -96 | 46 | -2.09 |
| BTC Daily | lstm | LSTM | 708 | 307 | 401 | 43.36% | 38.75% | 42.50% | 6.64 pp | -94 | 42 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 534 | 213 | 321 | 39.89% | 39.17% | 39.38% | 10.11 pp | -108 | 46 | -2.35 |
| BTC Daily | rf | RandomForest | 708 | 304 | 404 | 42.94% | 41.67% | 43.54% | 7.06 pp | -100 | 42 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 534 | 212 | 322 | 39.70% | 37.08% | 40.21% | 10.30 pp | -110 | 46 | -2.39 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 885 | 379 | 506 | 42.82% | 38.75% | 42.50% | 7.18 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 885 | 375 | 510 | 42.37% | 42.50% | 42.50% | 7.63 pp | -135 | 47 | -2.87 |
| BTC Daily | xgb | XGBoost | 718 | 283 | 435 | 39.42% | 34.17% | 39.38% | 10.58 pp | -152 | 42 | -3.62 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 885 | 420 | 465 | 47.46% | 49.17% | 47.92% | 2.54 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 885 | 420 | 465 | 47.46% | 48.75% | 47.92% | 2.54 pp | -45 | 47 | -0.96 |
| BTC Hourly | nn | NN | 885 | 396 | 489 | 44.75% | 45.42% | 43.33% | 5.25 pp | -93 | 47 | -1.98 |
| BTC Hourly | rf | RandomForest | 885 | 395 | 490 | 44.63% | 44.58% | 44.38% | 5.37 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 885 | 379 | 506 | 42.82% | 38.75% | 42.50% | 7.18 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 885 | 375 | 510 | 42.37% | 42.50% | 42.50% | 7.63 pp | -135 | 47 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 708 | 346 | 362 | 48.87% | 47.50% | 48.75% | 1.13 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 708 | 339 | 369 | 47.88% | 47.08% | 50.00% | 2.12 pp | -30 | 42 | -0.71 |
| BTC Daily | nn | NN | 708 | 328 | 380 | 46.33% | 43.33% | 47.92% | 3.67 pp | -52 | 42 | -1.24 |
| BTC Daily | lstm | LSTM | 708 | 307 | 401 | 43.36% | 38.75% | 42.50% | 6.64 pp | -94 | 42 | -2.24 |
| BTC Daily | rf | RandomForest | 708 | 304 | 404 | 42.94% | 41.67% | 43.54% | 7.06 pp | -100 | 42 | -2.38 |
| BTC Daily | xgb | XGBoost | 718 | 283 | 435 | 39.42% | 34.17% | 39.38% | 10.58 pp | -152 | 42 | -3.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 480 | 231 | 249 | 48.12% | 43.75% | 48.12% | 1.87 pp | -18 | 46 | -0.39 |
| BTC Market Hours | nn | NN | 480 | 226 | 254 | 47.08% | 48.33% | 47.08% | 2.92 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 480 | 224 | 256 | 46.67% | 41.67% | 46.67% | 3.33 pp | -32 | 46 | -0.70 |
| BTC Market Hours | lstm | LSTM | 480 | 207 | 273 | 43.12% | 41.67% | 43.12% | 6.87 pp | -66 | 46 | -1.43 |
| BTC Market Hours | rf | RandomForest | 480 | 205 | 275 | 42.71% | 42.08% | 42.71% | 7.29 pp | -70 | 46 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 480 | 195 | 285 | 40.62% | 39.58% | 40.62% | 9.38 pp | -90 | 46 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 534 | 248 | 286 | 46.44% | 49.58% | 47.50% | 3.56 pp | -38 | 46 | -0.83 |
| BTC Market Hours Daily | nn | NN | 534 | 245 | 289 | 45.88% | 43.75% | 46.67% | 4.12 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 534 | 243 | 291 | 45.51% | 47.08% | 46.46% | 4.49 pp | -48 | 46 | -1.04 |
| BTC Market Hours Daily | rf | RandomForest | 534 | 219 | 315 | 41.01% | 40.83% | 41.25% | 8.99 pp | -96 | 46 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 534 | 213 | 321 | 39.89% | 39.17% | 39.38% | 10.11 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | lstm | LSTM | 534 | 212 | 322 | 39.70% | 37.08% | 40.21% | 10.30 pp | -110 | 46 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
