# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T02:41:25.070951+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1302 | 1014 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1177 | 812 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 910 | 574 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 912 | 628 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 217 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 18:00:00+00:00 | 217 | 73 | 144 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 574 | 280 | 294 | 48.78% | 47.92% | 48.12% | 1.22 pp | -14 | 54 | -0.26 |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| BTC Market Hours | nn | NN | 574 | 274 | 300 | 47.74% | 52.08% | 49.38% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 574 | 270 | 304 | 47.04% | 46.67% | 46.88% | 2.96 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| BTC Market Hours Daily | nn | NN | 628 | 293 | 335 | 46.66% | 47.50% | 47.92% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 628 | 293 | 335 | 46.66% | 49.17% | 47.29% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 628 | 292 | 336 | 46.50% | 48.33% | 46.88% | 3.50 pp | -44 | 53 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 980 | 468 | 512 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 802 | 372 | 430 | 46.38% | 44.58% | 44.58% | 3.62 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| BTC Market Hours | lstm | LSTM | 574 | 249 | 325 | 43.38% | 43.33% | 43.75% | 6.62 pp | -76 | 54 | -1.41 |
| BTC Hourly | transformer | Transformer | 980 | 454 | 526 | 46.33% | 43.33% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Market Hours | rf | RandomForest | 574 | 248 | 326 | 43.21% | 45.42% | 43.75% | 6.79 pp | -78 | 54 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 574 | 248 | 326 | 43.21% | 47.08% | 43.54% | 6.79 pp | -78 | 54 | -1.44 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 628 | 262 | 366 | 41.72% | 43.33% | 40.62% | 8.28 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 628 | 260 | 368 | 41.40% | 44.58% | 41.04% | 8.60 pp | -108 | 53 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 628 | 256 | 372 | 40.76% | 41.67% | 40.00% | 9.24 pp | -116 | 53 | -2.19 |
| BTC Hourly | rf | RandomForest | 980 | 433 | 547 | 44.18% | 41.67% | 42.71% | 5.82 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 980 | 432 | 548 | 44.08% | 41.25% | 42.29% | 5.92 pp | -116 | 51 | -2.27 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |
| BTC Daily | lstm | LSTM | 802 | 338 | 464 | 42.14% | 35.00% | 40.21% | 7.86 pp | -126 | 46 | -2.74 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 980 | 417 | 563 | 42.55% | 37.50% | 41.04% | 7.45 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 802 | 333 | 469 | 41.52% | 37.08% | 40.83% | 8.48 pp | -136 | 46 | -2.96 |
| BTC Hourly | xgb | XGBoost | 980 | 405 | 575 | 41.33% | 36.25% | 39.17% | 8.67 pp | -170 | 51 | -3.33 |
| BTC Daily | xgb | XGBoost | 812 | 315 | 497 | 38.79% | 35.00% | 35.42% | 11.21 pp | -182 | 46 | -3.96 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 980 | 468 | 512 | 47.76% | 50.83% | 46.88% | 2.24 pp | -44 | 51 | -0.86 |
| BTC Hourly | transformer | Transformer | 980 | 454 | 526 | 46.33% | 43.33% | 43.54% | 3.67 pp | -72 | 51 | -1.41 |
| BTC Hourly | rf | RandomForest | 980 | 433 | 547 | 44.18% | 41.67% | 42.71% | 5.82 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 980 | 432 | 548 | 44.08% | 41.25% | 42.29% | 5.92 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 980 | 417 | 563 | 42.55% | 37.50% | 41.04% | 7.45 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 980 | 405 | 575 | 41.33% | 36.25% | 39.17% | 8.67 pp | -170 | 51 | -3.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| BTC Daily | nn | NN | 802 | 372 | 430 | 46.38% | 44.58% | 44.58% | 3.62 pp | -58 | 46 | -1.26 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 802 | 338 | 464 | 42.14% | 35.00% | 40.21% | 7.86 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 802 | 333 | 469 | 41.52% | 37.08% | 40.83% | 8.48 pp | -136 | 46 | -2.96 |
| BTC Daily | xgb | XGBoost | 812 | 315 | 497 | 38.79% | 35.00% | 35.42% | 11.21 pp | -182 | 46 | -3.96 |

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
| BTC Market Hours Daily | nn | NN | 628 | 293 | 335 | 46.66% | 47.50% | 47.92% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 628 | 293 | 335 | 46.66% | 49.17% | 47.29% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 628 | 292 | 336 | 46.50% | 48.33% | 46.88% | 3.50 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 628 | 262 | 366 | 41.72% | 43.33% | 40.62% | 8.28 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 628 | 260 | 368 | 41.40% | 44.58% | 41.04% | 8.60 pp | -108 | 53 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 628 | 256 | 372 | 40.76% | 41.67% | 40.00% | 9.24 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 217 | 101 | 116 | 46.54% | 46.54% | 46.54% | 3.46 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 217 | 96 | 121 | 44.24% | 44.24% | 44.24% | 5.76 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 217 | 95 | 122 | 43.78% | 43.78% | 43.78% | 6.22 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 14 | -2.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
