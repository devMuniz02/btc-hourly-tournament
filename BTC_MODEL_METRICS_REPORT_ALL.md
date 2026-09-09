# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T12:31:12.903560+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1308 | 1020 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1184 | 819 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 917 | 581 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 919 | 635 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T21:00:00+00:00 | 223 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T21:00:00+00:00 | 223 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T21:00:00+00:00 | 223 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T21:00:00+00:00 | 224 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 581 | 282 | 299 | 48.54% | 47.08% | 47.50% | 1.46 pp | -17 | 54 | -0.31 |
| BTC Market Hours | nn | NN | 581 | 280 | 301 | 48.19% | 52.92% | 49.79% | 1.81 pp | -21 | 54 | -0.39 |
| BTC Market Hours | transformer | Transformer | 581 | 275 | 306 | 47.33% | 47.92% | 47.08% | 2.67 pp | -31 | 54 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 809 | 390 | 419 | 48.21% | 46.67% | 47.29% | 1.79 pp | -29 | 47 | -0.62 |
| BTC Market Hours Daily | nn | NN | 635 | 299 | 336 | 47.09% | 48.33% | 48.33% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 635 | 299 | 336 | 47.09% | 50.42% | 47.50% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 635 | 297 | 338 | 46.77% | 48.75% | 47.08% | 3.23 pp | -41 | 54 | -0.76 |
| Consolidated Hourly | rf | RandomForest | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 14 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 986 | 470 | 516 | 47.67% | 50.42% | 46.67% | 2.33 pp | -46 | 51 | -0.90 |
| BTC Daily | nn | NN | 809 | 376 | 433 | 46.48% | 44.17% | 45.42% | 3.52 pp | -57 | 47 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Daily | transformer | Transformer | 809 | 375 | 434 | 46.35% | 40.00% | 46.46% | 3.65 pp | -59 | 47 | -1.26 |
| BTC Hourly | transformer | Transformer | 986 | 456 | 530 | 46.25% | 43.75% | 43.75% | 3.75 pp | -74 | 51 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 223 | 101 | 122 | 45.29% | 45.29% | 45.29% | 4.71 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 101 | 122 | 45.29% | 45.29% | 45.29% | 4.71 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours | rf | RandomForest | 581 | 250 | 331 | 43.03% | 44.17% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 581 | 250 | 331 | 43.03% | 45.83% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | lstm | LSTM | 581 | 249 | 332 | 42.86% | 41.67% | 43.12% | 7.14 pp | -83 | 54 | -1.54 |
| Consolidated Hourly | xgb | XGBoost | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 635 | 264 | 371 | 41.57% | 42.92% | 40.62% | 8.43 pp | -107 | 54 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 635 | 263 | 372 | 41.42% | 44.58% | 40.83% | 8.58 pp | -109 | 54 | -2.02 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 635 | 259 | 376 | 40.79% | 41.25% | 40.21% | 9.21 pp | -117 | 54 | -2.17 |
| BTC Hourly | rf | RandomForest | 986 | 436 | 550 | 44.22% | 42.08% | 42.92% | 5.78 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 986 | 435 | 551 | 44.12% | 42.08% | 42.29% | 5.88 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | transformer | Transformer | 223 | 95 | 128 | 42.60% | 42.60% | 42.60% | 7.40 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 95 | 128 | 42.60% | 42.60% | 42.60% | 7.40 pp | -33 | 14 | -2.36 |
| Consolidated Hourly | nn | NN | 223 | 94 | 129 | 42.15% | 42.15% | 42.15% | 7.85 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 94 | 129 | 42.15% | 42.15% | 42.15% | 7.85 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 809 | 341 | 468 | 42.15% | 35.00% | 40.62% | 7.85 pp | -127 | 47 | -2.70 |
| BTC Hourly | lstm | LSTM | 986 | 420 | 566 | 42.60% | 37.92% | 40.83% | 7.40 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 809 | 336 | 473 | 41.53% | 36.67% | 41.46% | 8.47 pp | -137 | 47 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 986 | 406 | 580 | 41.18% | 35.42% | 38.75% | 8.82 pp | -174 | 51 | -3.41 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 819 | 320 | 499 | 39.07% | 35.00% | 35.83% | 10.93 pp | -179 | 47 | -3.81 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 986 | 470 | 516 | 47.67% | 50.42% | 46.67% | 2.33 pp | -46 | 51 | -0.90 |
| BTC Hourly | transformer | Transformer | 986 | 456 | 530 | 46.25% | 43.75% | 43.75% | 3.75 pp | -74 | 51 | -1.45 |
| BTC Hourly | rf | RandomForest | 986 | 436 | 550 | 44.22% | 42.08% | 42.92% | 5.78 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 986 | 435 | 551 | 44.12% | 42.08% | 42.29% | 5.88 pp | -116 | 51 | -2.27 |
| BTC Hourly | lstm | LSTM | 986 | 420 | 566 | 42.60% | 37.92% | 40.83% | 7.40 pp | -146 | 51 | -2.86 |
| BTC Hourly | xgb | XGBoost | 986 | 406 | 580 | 41.18% | 35.42% | 38.75% | 8.82 pp | -174 | 51 | -3.41 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 809 | 390 | 419 | 48.21% | 46.67% | 47.29% | 1.79 pp | -29 | 47 | -0.62 |
| BTC Daily | nn | NN | 809 | 376 | 433 | 46.48% | 44.17% | 45.42% | 3.52 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 809 | 375 | 434 | 46.35% | 40.00% | 46.46% | 3.65 pp | -59 | 47 | -1.26 |
| BTC Daily | lstm | LSTM | 809 | 341 | 468 | 42.15% | 35.00% | 40.62% | 7.85 pp | -127 | 47 | -2.70 |
| BTC Daily | rf | RandomForest | 809 | 336 | 473 | 41.53% | 36.67% | 41.46% | 8.47 pp | -137 | 47 | -2.91 |
| BTC Daily | xgb | XGBoost | 819 | 320 | 499 | 39.07% | 35.00% | 35.83% | 10.93 pp | -179 | 47 | -3.81 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 581 | 282 | 299 | 48.54% | 47.08% | 47.50% | 1.46 pp | -17 | 54 | -0.31 |
| BTC Market Hours | nn | NN | 581 | 280 | 301 | 48.19% | 52.92% | 49.79% | 1.81 pp | -21 | 54 | -0.39 |
| BTC Market Hours | transformer | Transformer | 581 | 275 | 306 | 47.33% | 47.92% | 47.08% | 2.67 pp | -31 | 54 | -0.57 |
| BTC Market Hours | rf | RandomForest | 581 | 250 | 331 | 43.03% | 44.17% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 581 | 250 | 331 | 43.03% | 45.83% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | lstm | LSTM | 581 | 249 | 332 | 42.86% | 41.67% | 43.12% | 7.14 pp | -83 | 54 | -1.54 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 635 | 299 | 336 | 47.09% | 48.33% | 48.33% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 635 | 299 | 336 | 47.09% | 50.42% | 47.50% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 635 | 297 | 338 | 46.77% | 48.75% | 47.08% | 3.23 pp | -41 | 54 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 635 | 264 | 371 | 41.57% | 42.92% | 40.62% | 8.43 pp | -107 | 54 | -1.98 |
| BTC Market Hours Daily | xgb | XGBoost | 635 | 263 | 372 | 41.42% | 44.58% | 40.83% | 8.58 pp | -109 | 54 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 635 | 259 | 376 | 40.79% | 41.25% | 40.21% | 9.21 pp | -117 | 54 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 223 | 101 | 122 | 45.29% | 45.29% | 45.29% | 4.71 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 223 | 95 | 128 | 42.60% | 42.60% | 42.60% | 7.40 pp | -33 | 14 | -2.36 |
| Consolidated Hourly | nn | NN | 223 | 94 | 129 | 42.15% | 42.15% | 42.15% | 7.85 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 101 | 122 | 45.29% | 45.29% | 45.29% | 4.71 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 95 | 128 | 42.60% | 42.60% | 42.60% | 7.40 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 94 | 129 | 42.15% | 42.15% | 42.15% | 7.85 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
