# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T08:28:58.155914+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1208 | 920 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1084 | 719 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 739 | 481 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 741 | 535 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 131 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 709 | 366 | 343 | 51.62% | 49.58% | 51.67% | 1.62 pp | 23 | 42 | 0.55 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 481 | 231 | 250 | 48.02% | 43.75% | 48.12% | 1.98 pp | -19 | 46 | -0.41 |
| BTC Market Hours | nn | NN | 481 | 227 | 254 | 47.19% | 48.75% | 47.29% | 2.81 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 481 | 225 | 256 | 46.78% | 42.08% | 46.88% | 3.22 pp | -31 | 46 | -0.67 |
| BTC Daily | nn | NN | 709 | 338 | 371 | 47.67% | 46.67% | 48.75% | 2.33 pp | -33 | 42 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 535 | 248 | 287 | 46.36% | 49.58% | 47.29% | 3.64 pp | -39 | 46 | -0.85 |
| BTC Daily | transformer | Transformer | 709 | 335 | 374 | 47.25% | 46.25% | 49.38% | 2.75 pp | -39 | 42 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 886 | 421 | 465 | 47.52% | 49.17% | 47.92% | 2.48 pp | -44 | 47 | -0.94 |
| BTC Hourly | transformer | Transformer | 886 | 421 | 465 | 47.52% | 48.75% | 47.92% | 2.48 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | nn | NN | 535 | 245 | 290 | 45.79% | 43.75% | 46.67% | 4.21 pp | -45 | 46 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 535 | 244 | 291 | 45.61% | 47.50% | 46.46% | 4.39 pp | -47 | 46 | -1.02 |
| BTC Market Hours | lstm | LSTM | 481 | 208 | 273 | 43.24% | 42.08% | 43.12% | 6.76 pp | -65 | 46 | -1.41 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| BTC Market Hours | rf | RandomForest | 481 | 205 | 276 | 42.62% | 42.08% | 42.71% | 7.38 pp | -71 | 46 | -1.54 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| BTC Daily | lstm | LSTM | 709 | 320 | 389 | 45.13% | 39.17% | 44.58% | 4.87 pp | -69 | 42 | -1.64 |
| BTC Daily | rf | RandomForest | 709 | 318 | 391 | 44.85% | 42.50% | 44.79% | 5.15 pp | -73 | 42 | -1.74 |
| BTC Hourly | nn | NN | 886 | 397 | 489 | 44.81% | 45.83% | 43.33% | 5.19 pp | -92 | 47 | -1.96 |
| BTC Market Hours | xgb | XGBoost | 481 | 195 | 286 | 40.54% | 39.58% | 40.62% | 9.46 pp | -91 | 46 | -1.98 |
| BTC Hourly | rf | RandomForest | 886 | 396 | 490 | 44.70% | 45.00% | 44.38% | 5.30 pp | -94 | 47 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 535 | 220 | 315 | 41.12% | 41.25% | 41.25% | 8.88 pp | -95 | 46 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 535 | 213 | 322 | 39.81% | 39.17% | 39.38% | 10.19 pp | -109 | 46 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 535 | 212 | 323 | 39.63% | 37.08% | 40.00% | 10.37 pp | -111 | 46 | -2.41 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 886 | 380 | 506 | 42.89% | 38.75% | 42.50% | 7.11 pp | -126 | 47 | -2.68 |
| BTC Hourly | xgb | XGBoost | 886 | 375 | 511 | 42.33% | 42.50% | 42.29% | 7.67 pp | -136 | 47 | -2.89 |
| BTC Daily | xgb | XGBoost | 719 | 288 | 431 | 40.06% | 36.25% | 39.58% | 9.94 pp | -143 | 42 | -3.40 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 886 | 421 | 465 | 47.52% | 49.17% | 47.92% | 2.48 pp | -44 | 47 | -0.94 |
| BTC Hourly | transformer | Transformer | 886 | 421 | 465 | 47.52% | 48.75% | 47.92% | 2.48 pp | -44 | 47 | -0.94 |
| BTC Hourly | nn | NN | 886 | 397 | 489 | 44.81% | 45.83% | 43.33% | 5.19 pp | -92 | 47 | -1.96 |
| BTC Hourly | rf | RandomForest | 886 | 396 | 490 | 44.70% | 45.00% | 44.38% | 5.30 pp | -94 | 47 | -2.00 |
| BTC Hourly | lstm | LSTM | 886 | 380 | 506 | 42.89% | 38.75% | 42.50% | 7.11 pp | -126 | 47 | -2.68 |
| BTC Hourly | xgb | XGBoost | 886 | 375 | 511 | 42.33% | 42.50% | 42.29% | 7.67 pp | -136 | 47 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 709 | 366 | 343 | 51.62% | 49.58% | 51.67% | 1.62 pp | 23 | 42 | 0.55 |
| BTC Daily | nn | NN | 709 | 338 | 371 | 47.67% | 46.67% | 48.75% | 2.33 pp | -33 | 42 | -0.79 |
| BTC Daily | transformer | Transformer | 709 | 335 | 374 | 47.25% | 46.25% | 49.38% | 2.75 pp | -39 | 42 | -0.93 |
| BTC Daily | lstm | LSTM | 709 | 320 | 389 | 45.13% | 39.17% | 44.58% | 4.87 pp | -69 | 42 | -1.64 |
| BTC Daily | rf | RandomForest | 709 | 318 | 391 | 44.85% | 42.50% | 44.79% | 5.15 pp | -73 | 42 | -1.74 |
| BTC Daily | xgb | XGBoost | 719 | 288 | 431 | 40.06% | 36.25% | 39.58% | 9.94 pp | -143 | 42 | -3.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 481 | 231 | 250 | 48.02% | 43.75% | 48.12% | 1.98 pp | -19 | 46 | -0.41 |
| BTC Market Hours | nn | NN | 481 | 227 | 254 | 47.19% | 48.75% | 47.29% | 2.81 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 481 | 225 | 256 | 46.78% | 42.08% | 46.88% | 3.22 pp | -31 | 46 | -0.67 |
| BTC Market Hours | lstm | LSTM | 481 | 208 | 273 | 43.24% | 42.08% | 43.12% | 6.76 pp | -65 | 46 | -1.41 |
| BTC Market Hours | rf | RandomForest | 481 | 205 | 276 | 42.62% | 42.08% | 42.71% | 7.38 pp | -71 | 46 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 481 | 195 | 286 | 40.54% | 39.58% | 40.62% | 9.46 pp | -91 | 46 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 535 | 248 | 287 | 46.36% | 49.58% | 47.29% | 3.64 pp | -39 | 46 | -0.85 |
| BTC Market Hours Daily | nn | NN | 535 | 245 | 290 | 45.79% | 43.75% | 46.67% | 4.21 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 535 | 244 | 291 | 45.61% | 47.50% | 46.46% | 4.39 pp | -47 | 46 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 535 | 220 | 315 | 41.12% | 41.25% | 41.25% | 8.88 pp | -95 | 46 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 535 | 213 | 322 | 39.81% | 39.17% | 39.38% | 10.19 pp | -109 | 46 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 535 | 212 | 323 | 39.63% | 37.08% | 40.00% | 10.37 pp | -111 | 46 | -2.41 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
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
