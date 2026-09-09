# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T11:42:10.041962+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 918 | 634 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 581 | 282 | 299 | 48.54% | 47.08% | 47.50% | 1.46 pp | -17 | 54 | -0.31 |
| BTC Market Hours | nn | NN | 581 | 280 | 301 | 48.19% | 52.92% | 49.79% | 1.81 pp | -21 | 54 | -0.39 |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 581 | 275 | 306 | 47.33% | 47.92% | 47.08% | 2.67 pp | -31 | 54 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 809 | 390 | 419 | 48.21% | 46.67% | 47.29% | 1.79 pp | -29 | 47 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 634 | 299 | 335 | 47.16% | 50.42% | 47.71% | 2.84 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | nn | NN | 634 | 298 | 336 | 47.00% | 47.92% | 48.12% | 3.00 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 634 | 296 | 338 | 46.69% | 48.33% | 46.88% | 3.31 pp | -42 | 54 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 986 | 470 | 516 | 47.67% | 50.42% | 46.67% | 2.33 pp | -46 | 51 | -0.90 |
| BTC Daily | nn | NN | 809 | 376 | 433 | 46.48% | 44.17% | 45.42% | 3.52 pp | -57 | 47 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Daily | transformer | Transformer | 809 | 375 | 434 | 46.35% | 40.00% | 46.46% | 3.65 pp | -59 | 47 | -1.26 |
| BTC Hourly | transformer | Transformer | 986 | 456 | 530 | 46.25% | 43.75% | 43.75% | 3.75 pp | -74 | 51 | -1.45 |
| BTC Market Hours | rf | RandomForest | 581 | 250 | 331 | 43.03% | 44.17% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 581 | 250 | 331 | 43.03% | 45.83% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | lstm | LSTM | 581 | 249 | 332 | 42.86% | 41.67% | 43.12% | 7.14 pp | -83 | 54 | -1.54 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 634 | 263 | 371 | 41.48% | 42.50% | 40.42% | 8.52 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 634 | 262 | 372 | 41.32% | 44.58% | 40.62% | 8.68 pp | -110 | 54 | -2.04 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 634 | 258 | 376 | 40.69% | 40.83% | 40.00% | 9.31 pp | -118 | 54 | -2.19 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 986 | 436 | 550 | 44.22% | 42.08% | 42.92% | 5.78 pp | -114 | 51 | -2.24 |
| BTC Hourly | nn | NN | 986 | 435 | 551 | 44.12% | 42.08% | 42.29% | 5.88 pp | -116 | 51 | -2.27 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Daily | lstm | LSTM | 809 | 341 | 468 | 42.15% | 35.00% | 40.62% | 7.85 pp | -127 | 47 | -2.70 |
| BTC Hourly | lstm | LSTM | 986 | 420 | 566 | 42.60% | 37.92% | 40.83% | 7.40 pp | -146 | 51 | -2.86 |
| BTC Daily | rf | RandomForest | 809 | 336 | 473 | 41.53% | 36.67% | 41.46% | 8.47 pp | -137 | 47 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 986 | 406 | 580 | 41.18% | 35.42% | 38.75% | 8.82 pp | -174 | 51 | -3.41 |
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
| BTC Market Hours Daily | transformer | Transformer | 634 | 299 | 335 | 47.16% | 50.42% | 47.71% | 2.84 pp | -36 | 54 | -0.67 |
| BTC Market Hours Daily | nn | NN | 634 | 298 | 336 | 47.00% | 47.92% | 48.12% | 3.00 pp | -38 | 54 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 634 | 296 | 338 | 46.69% | 48.33% | 46.88% | 3.31 pp | -42 | 54 | -0.78 |
| BTC Market Hours Daily | rf | RandomForest | 634 | 263 | 371 | 41.48% | 42.50% | 40.42% | 8.52 pp | -108 | 54 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 634 | 262 | 372 | 41.32% | 44.58% | 40.62% | 8.68 pp | -110 | 54 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 634 | 258 | 376 | 40.69% | 40.83% | 40.00% | 9.31 pp | -118 | 54 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
