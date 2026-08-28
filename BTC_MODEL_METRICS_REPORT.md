# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T16:00:48.094286+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 822 | 297 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 986 | 621 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 567 | 383 | 183 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 569 | 437 | 130 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 1 | 43 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 1 | 43 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 383 | 188 | 195 | 49.09% | 47.50% | 49.09% | 0.91 pp | -7 | 39 | -0.18 |
| BTC Daily | transformer | Transformer | 611 | 300 | 311 | 49.10% | 49.58% | 50.21% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 611 | 299 | 312 | 48.94% | 47.08% | 50.00% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 383 | 178 | 205 | 46.48% | 43.33% | 46.48% | 3.52 pp | -27 | 39 | -0.69 |
| BTC Market Hours | nn | NN | 383 | 177 | 206 | 46.21% | 48.75% | 46.21% | 3.79 pp | -29 | 39 | -0.74 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 437 | 200 | 237 | 45.77% | 47.50% | 45.77% | 4.23 pp | -37 | 39 | -0.95 |
| BTC Daily | nn | NN | 611 | 287 | 324 | 46.97% | 44.17% | 48.54% | 3.03 pp | -37 | 38 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 788 | 373 | 415 | 47.34% | 45.00% | 47.50% | 2.66 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 437 | 199 | 238 | 45.54% | 45.83% | 45.54% | 4.46 pp | -39 | 39 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 437 | 198 | 239 | 45.31% | 46.25% | 45.31% | 4.69 pp | -41 | 39 | -1.05 |
| BTC Hourly | transformer | Transformer | 788 | 370 | 418 | 46.95% | 43.75% | 46.25% | 3.05 pp | -48 | 43 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 383 | 164 | 219 | 42.82% | 43.33% | 42.82% | 7.18 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 383 | 163 | 220 | 42.56% | 40.83% | 42.56% | 7.44 pp | -57 | 39 | -1.46 |
| BTC Daily | lstm | LSTM | 611 | 273 | 338 | 44.68% | 44.17% | 44.79% | 5.32 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 788 | 355 | 433 | 45.05% | 40.83% | 45.83% | 4.95 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 383 | 154 | 229 | 40.21% | 38.33% | 40.21% | 9.79 pp | -75 | 39 | -1.92 |
| BTC Hourly | rf | RandomForest | 788 | 350 | 438 | 44.42% | 42.50% | 43.96% | 5.58 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 437 | 177 | 260 | 40.50% | 38.75% | 40.50% | 9.50 pp | -83 | 39 | -2.13 |
| BTC Hourly | lstm | LSTM | 788 | 347 | 441 | 44.04% | 43.75% | 45.42% | 5.96 pp | -94 | 43 | -2.19 |
| BTC Daily | rf | RandomForest | 611 | 263 | 348 | 43.04% | 43.33% | 43.75% | 6.96 pp | -85 | 38 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 437 | 172 | 265 | 39.36% | 37.08% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 437 | 172 | 265 | 39.36% | 37.92% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| Consolidated Hourly | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |
| BTC Hourly | xgb | XGBoost | 788 | 336 | 452 | 42.64% | 39.58% | 44.17% | 7.36 pp | -116 | 43 | -2.70 |
| BTC Daily | xgb | XGBoost | 621 | 247 | 374 | 39.77% | 33.75% | 40.00% | 10.23 pp | -127 | 38 | -3.34 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 788 | 373 | 415 | 47.34% | 45.00% | 47.50% | 2.66 pp | -42 | 43 | -0.98 |
| BTC Hourly | transformer | Transformer | 788 | 370 | 418 | 46.95% | 43.75% | 46.25% | 3.05 pp | -48 | 43 | -1.12 |
| BTC Hourly | nn | NN | 788 | 355 | 433 | 45.05% | 40.83% | 45.83% | 4.95 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 788 | 350 | 438 | 44.42% | 42.50% | 43.96% | 5.58 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 788 | 347 | 441 | 44.04% | 43.75% | 45.42% | 5.96 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 788 | 336 | 452 | 42.64% | 39.58% | 44.17% | 7.36 pp | -116 | 43 | -2.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 611 | 300 | 311 | 49.10% | 49.58% | 50.21% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 611 | 299 | 312 | 48.94% | 47.08% | 50.00% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 611 | 287 | 324 | 46.97% | 44.17% | 48.54% | 3.03 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 611 | 273 | 338 | 44.68% | 44.17% | 44.79% | 5.32 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 611 | 263 | 348 | 43.04% | 43.33% | 43.75% | 6.96 pp | -85 | 38 | -2.24 |
| BTC Daily | xgb | XGBoost | 621 | 247 | 374 | 39.77% | 33.75% | 40.00% | 10.23 pp | -127 | 38 | -3.34 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 383 | 188 | 195 | 49.09% | 47.50% | 49.09% | 0.91 pp | -7 | 39 | -0.18 |
| BTC Market Hours | transformer | Transformer | 383 | 178 | 205 | 46.48% | 43.33% | 46.48% | 3.52 pp | -27 | 39 | -0.69 |
| BTC Market Hours | nn | NN | 383 | 177 | 206 | 46.21% | 48.75% | 46.21% | 3.79 pp | -29 | 39 | -0.74 |
| BTC Market Hours | lstm | LSTM | 383 | 164 | 219 | 42.82% | 43.33% | 42.82% | 7.18 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 383 | 163 | 220 | 42.56% | 40.83% | 42.56% | 7.44 pp | -57 | 39 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 383 | 154 | 229 | 40.21% | 38.33% | 40.21% | 9.79 pp | -75 | 39 | -1.92 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 437 | 200 | 237 | 45.77% | 47.50% | 45.77% | 4.23 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 437 | 199 | 238 | 45.54% | 45.83% | 45.54% | 4.46 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 437 | 198 | 239 | 45.31% | 46.25% | 45.31% | 4.69 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 437 | 177 | 260 | 40.50% | 38.75% | 40.50% | 9.50 pp | -83 | 39 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 437 | 172 | 265 | 39.36% | 37.08% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 437 | 172 | 265 | 39.36% | 37.92% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
