# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T06:10:41.804139+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1223 | 935 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1099 | 734 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 767 | 496 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 768 | 549 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 496 | 240 | 256 | 48.39% | 44.58% | 48.33% | 1.61 pp | -16 | 48 | -0.33 |
| BTC Market Hours | nn | NN | 496 | 234 | 262 | 47.18% | 50.00% | 47.71% | 2.82 pp | -28 | 48 | -0.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 724 | 349 | 375 | 48.20% | 46.25% | 47.71% | 1.80 pp | -26 | 43 | -0.60 |
| BTC Market Hours | transformer | Transformer | 496 | 233 | 263 | 46.98% | 44.17% | 47.71% | 3.02 pp | -30 | 48 | -0.62 |
| BTC Daily | transformer | Transformer | 724 | 346 | 378 | 47.79% | 46.67% | 50.00% | 2.21 pp | -32 | 43 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 901 | 431 | 470 | 47.84% | 51.25% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 549 | 255 | 294 | 46.45% | 49.17% | 47.29% | 3.55 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 549 | 253 | 296 | 46.08% | 45.00% | 47.29% | 3.92 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 549 | 252 | 297 | 45.90% | 48.75% | 46.88% | 4.10 pp | -45 | 47 | -0.96 |
| BTC Hourly | transformer | Transformer | 901 | 427 | 474 | 47.39% | 48.33% | 46.88% | 2.61 pp | -47 | 48 | -0.98 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 724 | 336 | 388 | 46.41% | 45.00% | 47.71% | 3.59 pp | -52 | 43 | -1.21 |
| BTC Market Hours | lstm | LSTM | 496 | 214 | 282 | 43.15% | 40.83% | 43.12% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 496 | 214 | 282 | 43.15% | 43.75% | 43.54% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 48 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 549 | 229 | 320 | 41.71% | 42.50% | 41.25% | 8.29 pp | -91 | 47 | -1.94 |
| BTC Hourly | nn | NN | 901 | 401 | 500 | 44.51% | 44.17% | 42.29% | 5.49 pp | -99 | 48 | -2.06 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| BTC Hourly | rf | RandomForest | 901 | 400 | 501 | 44.40% | 44.17% | 43.96% | 5.60 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 549 | 221 | 328 | 40.26% | 38.33% | 40.62% | 9.74 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 549 | 220 | 329 | 40.07% | 41.25% | 39.38% | 9.93 pp | -109 | 47 | -2.32 |
| BTC Daily | lstm | LSTM | 724 | 312 | 412 | 43.09% | 37.08% | 41.67% | 6.91 pp | -100 | 43 | -2.33 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 724 | 310 | 414 | 42.82% | 41.25% | 43.75% | 7.18 pp | -104 | 43 | -2.42 |
| BTC Hourly | lstm | LSTM | 901 | 386 | 515 | 42.84% | 40.00% | 42.29% | 7.16 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 901 | 379 | 522 | 42.06% | 42.08% | 41.67% | 7.94 pp | -143 | 48 | -2.98 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 734 | 292 | 442 | 39.78% | 37.08% | 38.96% | 10.22 pp | -150 | 43 | -3.49 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 901 | 431 | 470 | 47.84% | 51.25% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 901 | 427 | 474 | 47.39% | 48.33% | 46.88% | 2.61 pp | -47 | 48 | -0.98 |
| BTC Hourly | nn | NN | 901 | 401 | 500 | 44.51% | 44.17% | 42.29% | 5.49 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 901 | 400 | 501 | 44.40% | 44.17% | 43.96% | 5.60 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 901 | 386 | 515 | 42.84% | 40.00% | 42.29% | 7.16 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 901 | 379 | 522 | 42.06% | 42.08% | 41.67% | 7.94 pp | -143 | 48 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 724 | 349 | 375 | 48.20% | 46.25% | 47.71% | 1.80 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 724 | 346 | 378 | 47.79% | 46.67% | 50.00% | 2.21 pp | -32 | 43 | -0.74 |
| BTC Daily | nn | NN | 724 | 336 | 388 | 46.41% | 45.00% | 47.71% | 3.59 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 724 | 312 | 412 | 43.09% | 37.08% | 41.67% | 6.91 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 724 | 310 | 414 | 42.82% | 41.25% | 43.75% | 7.18 pp | -104 | 43 | -2.42 |
| BTC Daily | xgb | XGBoost | 734 | 292 | 442 | 39.78% | 37.08% | 38.96% | 10.22 pp | -150 | 43 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 496 | 240 | 256 | 48.39% | 44.58% | 48.33% | 1.61 pp | -16 | 48 | -0.33 |
| BTC Market Hours | nn | NN | 496 | 234 | 262 | 47.18% | 50.00% | 47.71% | 2.82 pp | -28 | 48 | -0.58 |
| BTC Market Hours | transformer | Transformer | 496 | 233 | 263 | 46.98% | 44.17% | 47.71% | 3.02 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 496 | 214 | 282 | 43.15% | 40.83% | 43.12% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 496 | 214 | 282 | 43.15% | 43.75% | 43.54% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 549 | 255 | 294 | 46.45% | 49.17% | 47.29% | 3.55 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 549 | 253 | 296 | 46.08% | 45.00% | 47.29% | 3.92 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 549 | 252 | 297 | 45.90% | 48.75% | 46.88% | 4.10 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 549 | 229 | 320 | 41.71% | 42.50% | 41.25% | 8.29 pp | -91 | 47 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 549 | 221 | 328 | 40.26% | 38.33% | 40.62% | 9.74 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 549 | 220 | 329 | 40.07% | 41.25% | 39.38% | 9.93 pp | -109 | 47 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
