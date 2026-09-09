# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T22:26:06.514621+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1314 | 1026 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1190 | 825 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 933 | 587 | 345 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 935 | 641 | 292 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 587 | 285 | 302 | 48.55% | 47.92% | 47.71% | 1.45 pp | -17 | 55 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| BTC Market Hours | nn | NN | 587 | 283 | 304 | 48.21% | 52.50% | 49.79% | 1.79 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 587 | 277 | 310 | 47.19% | 47.50% | 46.67% | 2.81 pp | -33 | 55 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 815 | 392 | 423 | 48.10% | 45.42% | 46.88% | 1.90 pp | -31 | 47 | -0.66 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 641 | 301 | 340 | 46.96% | 49.58% | 47.50% | 3.04 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 641 | 300 | 341 | 46.80% | 48.33% | 48.12% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 641 | 300 | 341 | 46.80% | 48.75% | 47.50% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 992 | 471 | 521 | 47.48% | 49.17% | 46.25% | 2.52 pp | -50 | 51 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 815 | 380 | 435 | 46.63% | 45.00% | 45.42% | 3.37 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 815 | 378 | 437 | 46.38% | 40.00% | 46.04% | 3.62 pp | -59 | 47 | -1.26 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 992 | 461 | 531 | 46.47% | 44.58% | 44.17% | 3.53 pp | -70 | 51 | -1.37 |
| BTC Market Hours | rf | RandomForest | 587 | 251 | 336 | 42.76% | 43.33% | 43.12% | 7.24 pp | -85 | 55 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 587 | 251 | 336 | 42.76% | 44.17% | 43.12% | 7.24 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 587 | 250 | 337 | 42.59% | 42.08% | 42.50% | 7.41 pp | -87 | 55 | -1.58 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 641 | 266 | 375 | 41.50% | 42.50% | 41.04% | 8.50 pp | -109 | 54 | -2.02 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 641 | 264 | 377 | 41.19% | 42.92% | 40.62% | 8.81 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 641 | 263 | 378 | 41.03% | 42.08% | 41.04% | 8.97 pp | -115 | 54 | -2.13 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| BTC Hourly | nn | NN | 992 | 438 | 554 | 44.15% | 42.50% | 42.08% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 992 | 438 | 554 | 44.15% | 41.25% | 42.92% | 5.85 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| BTC Daily | lstm | LSTM | 815 | 345 | 470 | 42.33% | 36.25% | 40.62% | 7.67 pp | -125 | 47 | -2.66 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| BTC Hourly | lstm | LSTM | 992 | 421 | 571 | 42.44% | 36.67% | 40.21% | 7.56 pp | -150 | 51 | -2.94 |
| BTC Daily | rf | RandomForest | 815 | 338 | 477 | 41.47% | 37.50% | 40.83% | 8.53 pp | -139 | 47 | -2.96 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 992 | 407 | 585 | 41.03% | 34.58% | 38.33% | 8.97 pp | -178 | 51 | -3.49 |
| BTC Daily | xgb | XGBoost | 825 | 324 | 501 | 39.27% | 36.67% | 36.04% | 10.73 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 992 | 471 | 521 | 47.48% | 49.17% | 46.25% | 2.52 pp | -50 | 51 | -0.98 |
| BTC Hourly | transformer | Transformer | 992 | 461 | 531 | 46.47% | 44.58% | 44.17% | 3.53 pp | -70 | 51 | -1.37 |
| BTC Hourly | nn | NN | 992 | 438 | 554 | 44.15% | 42.50% | 42.08% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | rf | RandomForest | 992 | 438 | 554 | 44.15% | 41.25% | 42.92% | 5.85 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 992 | 421 | 571 | 42.44% | 36.67% | 40.21% | 7.56 pp | -150 | 51 | -2.94 |
| BTC Hourly | xgb | XGBoost | 992 | 407 | 585 | 41.03% | 34.58% | 38.33% | 8.97 pp | -178 | 51 | -3.49 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 815 | 392 | 423 | 48.10% | 45.42% | 46.88% | 1.90 pp | -31 | 47 | -0.66 |
| BTC Daily | nn | NN | 815 | 380 | 435 | 46.63% | 45.00% | 45.42% | 3.37 pp | -55 | 47 | -1.17 |
| BTC Daily | transformer | Transformer | 815 | 378 | 437 | 46.38% | 40.00% | 46.04% | 3.62 pp | -59 | 47 | -1.26 |
| BTC Daily | lstm | LSTM | 815 | 345 | 470 | 42.33% | 36.25% | 40.62% | 7.67 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 815 | 338 | 477 | 41.47% | 37.50% | 40.83% | 8.53 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 825 | 324 | 501 | 39.27% | 36.67% | 36.04% | 10.73 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 587 | 285 | 302 | 48.55% | 47.92% | 47.71% | 1.45 pp | -17 | 55 | -0.31 |
| BTC Market Hours | nn | NN | 587 | 283 | 304 | 48.21% | 52.50% | 49.79% | 1.79 pp | -21 | 55 | -0.38 |
| BTC Market Hours | transformer | Transformer | 587 | 277 | 310 | 47.19% | 47.50% | 46.67% | 2.81 pp | -33 | 55 | -0.60 |
| BTC Market Hours | rf | RandomForest | 587 | 251 | 336 | 42.76% | 43.33% | 43.12% | 7.24 pp | -85 | 55 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 587 | 251 | 336 | 42.76% | 44.17% | 43.12% | 7.24 pp | -85 | 55 | -1.55 |
| BTC Market Hours | lstm | LSTM | 587 | 250 | 337 | 42.59% | 42.08% | 42.50% | 7.41 pp | -87 | 55 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 641 | 301 | 340 | 46.96% | 49.58% | 47.50% | 3.04 pp | -39 | 54 | -0.72 |
| BTC Market Hours Daily | nn | NN | 641 | 300 | 341 | 46.80% | 48.33% | 48.12% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 641 | 300 | 341 | 46.80% | 48.75% | 47.50% | 3.20 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 641 | 266 | 375 | 41.50% | 42.50% | 41.04% | 8.50 pp | -109 | 54 | -2.02 |
| BTC Market Hours Daily | xgb | XGBoost | 641 | 264 | 377 | 41.19% | 42.92% | 40.62% | 8.81 pp | -113 | 54 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 641 | 263 | 378 | 41.03% | 42.08% | 41.04% | 8.97 pp | -115 | 54 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
