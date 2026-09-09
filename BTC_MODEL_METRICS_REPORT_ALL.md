# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T12:58:28.530318+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1309 | 1021 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1184 | 819 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 917 | 581 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 919 | 635 | 282 | 2 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 809 | 389 | 420 | 48.08% | 46.25% | 47.08% | 1.92 pp | -31 | 47 | -0.66 |
| BTC Market Hours Daily | nn | NN | 635 | 299 | 336 | 47.09% | 48.33% | 48.33% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 635 | 299 | 336 | 47.09% | 50.42% | 47.50% | 2.91 pp | -37 | 54 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 635 | 297 | 338 | 46.77% | 48.75% | 47.08% | 3.23 pp | -41 | 54 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 987 | 470 | 517 | 47.62% | 50.00% | 46.67% | 2.38 pp | -47 | 51 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Daily | nn | NN | 809 | 375 | 434 | 46.35% | 44.17% | 45.21% | 3.65 pp | -59 | 47 | -1.26 |
| BTC Daily | transformer | Transformer | 809 | 374 | 435 | 46.23% | 39.58% | 46.25% | 3.77 pp | -61 | 47 | -1.30 |
| BTC Hourly | transformer | Transformer | 987 | 457 | 530 | 46.30% | 44.17% | 43.75% | 3.70 pp | -73 | 51 | -1.43 |
| BTC Market Hours | rf | RandomForest | 581 | 250 | 331 | 43.03% | 44.17% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 581 | 250 | 331 | 43.03% | 45.83% | 43.33% | 6.97 pp | -81 | 54 | -1.50 |
| BTC Market Hours | lstm | LSTM | 581 | 249 | 332 | 42.86% | 41.67% | 43.12% | 7.14 pp | -83 | 54 | -1.54 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 635 | 264 | 371 | 41.57% | 42.92% | 40.62% | 8.43 pp | -107 | 54 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 635 | 263 | 372 | 41.42% | 44.58% | 40.83% | 8.58 pp | -109 | 54 | -2.02 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 635 | 259 | 376 | 40.79% | 41.25% | 40.21% | 9.21 pp | -117 | 54 | -2.17 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 987 | 436 | 551 | 44.17% | 41.67% | 42.92% | 5.83 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 987 | 435 | 552 | 44.07% | 41.67% | 42.08% | 5.93 pp | -117 | 51 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Daily | lstm | LSTM | 809 | 342 | 467 | 42.27% | 35.42% | 40.83% | 7.73 pp | -125 | 47 | -2.66 |
| BTC Hourly | lstm | LSTM | 987 | 420 | 567 | 42.55% | 37.92% | 40.62% | 7.45 pp | -147 | 51 | -2.88 |
| BTC Daily | rf | RandomForest | 809 | 335 | 474 | 41.41% | 36.67% | 41.25% | 8.59 pp | -139 | 47 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 987 | 406 | 581 | 41.13% | 35.00% | 38.75% | 8.87 pp | -175 | 51 | -3.43 |
| BTC Daily | xgb | XGBoost | 819 | 319 | 500 | 38.95% | 35.00% | 35.62% | 11.05 pp | -181 | 47 | -3.85 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 987 | 470 | 517 | 47.62% | 50.00% | 46.67% | 2.38 pp | -47 | 51 | -0.92 |
| BTC Hourly | transformer | Transformer | 987 | 457 | 530 | 46.30% | 44.17% | 43.75% | 3.70 pp | -73 | 51 | -1.43 |
| BTC Hourly | rf | RandomForest | 987 | 436 | 551 | 44.17% | 41.67% | 42.92% | 5.83 pp | -115 | 51 | -2.25 |
| BTC Hourly | nn | NN | 987 | 435 | 552 | 44.07% | 41.67% | 42.08% | 5.93 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 987 | 420 | 567 | 42.55% | 37.92% | 40.62% | 7.45 pp | -147 | 51 | -2.88 |
| BTC Hourly | xgb | XGBoost | 987 | 406 | 581 | 41.13% | 35.00% | 38.75% | 8.87 pp | -175 | 51 | -3.43 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 809 | 389 | 420 | 48.08% | 46.25% | 47.08% | 1.92 pp | -31 | 47 | -0.66 |
| BTC Daily | nn | NN | 809 | 375 | 434 | 46.35% | 44.17% | 45.21% | 3.65 pp | -59 | 47 | -1.26 |
| BTC Daily | transformer | Transformer | 809 | 374 | 435 | 46.23% | 39.58% | 46.25% | 3.77 pp | -61 | 47 | -1.30 |
| BTC Daily | lstm | LSTM | 809 | 342 | 467 | 42.27% | 35.42% | 40.83% | 7.73 pp | -125 | 47 | -2.66 |
| BTC Daily | rf | RandomForest | 809 | 335 | 474 | 41.41% | 36.67% | 41.25% | 8.59 pp | -139 | 47 | -2.96 |
| BTC Daily | xgb | XGBoost | 819 | 319 | 500 | 38.95% | 35.00% | 35.62% | 11.05 pp | -181 | 47 | -3.85 |

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
