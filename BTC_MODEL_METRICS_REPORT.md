# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T10:38:43.760700+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1242 | 954 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1118 | 753 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 799 | 515 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 801 | 569 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 161 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 161 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 161 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T14:00:00+00:00 | 162 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 515 | 249 | 266 | 48.35% | 45.42% | 48.33% | 1.65 pp | -17 | 49 | -0.35 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| BTC Market Hours | transformer | Transformer | 515 | 247 | 268 | 47.96% | 47.08% | 48.54% | 2.04 pp | -21 | 49 | -0.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 743 | 361 | 382 | 48.59% | 47.08% | 48.75% | 1.41 pp | -21 | 44 | -0.48 |
| BTC Market Hours Daily | transformer | Transformer | 569 | 271 | 298 | 47.63% | 51.25% | 48.75% | 2.37 pp | -27 | 49 | -0.55 |
| BTC Market Hours | nn | NN | 515 | 242 | 273 | 46.99% | 49.17% | 48.12% | 3.01 pp | -31 | 49 | -0.63 |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 743 | 354 | 389 | 47.64% | 45.83% | 49.58% | 2.36 pp | -35 | 44 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 920 | 439 | 481 | 47.72% | 48.75% | 47.08% | 2.28 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | nn | NN | 569 | 263 | 306 | 46.22% | 45.42% | 47.29% | 3.78 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 569 | 262 | 307 | 46.05% | 49.17% | 46.25% | 3.95 pp | -45 | 49 | -0.92 |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 920 | 435 | 485 | 47.28% | 47.92% | 46.04% | 2.72 pp | -50 | 48 | -1.04 |
| Consolidated Hourly | xgb | XGBoost | 161 | 74 | 87 | 45.96% | 45.96% | 45.96% | 4.04 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 74 | 87 | 45.96% | 45.96% | 45.96% | 4.04 pp | -13 | 12 | -1.08 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 743 | 343 | 400 | 46.16% | 42.92% | 46.67% | 3.84 pp | -57 | 44 | -1.30 |
| BTC Market Hours | lstm | LSTM | 515 | 224 | 291 | 43.50% | 42.50% | 43.96% | 6.50 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 515 | 222 | 293 | 43.11% | 44.58% | 43.54% | 6.89 pp | -71 | 49 | -1.45 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 515 | 212 | 303 | 41.17% | 42.92% | 42.08% | 8.83 pp | -91 | 49 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 569 | 237 | 332 | 41.65% | 42.50% | 40.83% | 8.35 pp | -95 | 49 | -1.94 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 920 | 409 | 511 | 44.46% | 43.75% | 43.96% | 5.54 pp | -102 | 48 | -2.12 |
| BTC Hourly | nn | NN | 920 | 408 | 512 | 44.35% | 42.50% | 42.29% | 5.65 pp | -104 | 48 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 569 | 231 | 338 | 40.60% | 40.00% | 40.62% | 9.40 pp | -107 | 49 | -2.18 |
| Consolidated Hourly | transformer | Transformer | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 569 | 227 | 342 | 39.89% | 40.83% | 38.96% | 10.11 pp | -115 | 49 | -2.35 |
| BTC Daily | lstm | LSTM | 743 | 318 | 425 | 42.80% | 36.25% | 40.83% | 7.20 pp | -107 | 44 | -2.43 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 743 | 313 | 430 | 42.13% | 38.75% | 42.50% | 7.87 pp | -117 | 44 | -2.66 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 920 | 393 | 527 | 42.72% | 38.33% | 41.67% | 7.28 pp | -134 | 48 | -2.79 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 920 | 384 | 536 | 41.74% | 38.75% | 40.00% | 8.26 pp | -152 | 48 | -3.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 753 | 297 | 456 | 39.44% | 36.25% | 37.71% | 10.56 pp | -159 | 44 | -3.61 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 920 | 439 | 481 | 47.72% | 48.75% | 47.08% | 2.28 pp | -42 | 48 | -0.88 |
| BTC Hourly | transformer | Transformer | 920 | 435 | 485 | 47.28% | 47.92% | 46.04% | 2.72 pp | -50 | 48 | -1.04 |
| BTC Hourly | rf | RandomForest | 920 | 409 | 511 | 44.46% | 43.75% | 43.96% | 5.54 pp | -102 | 48 | -2.12 |
| BTC Hourly | nn | NN | 920 | 408 | 512 | 44.35% | 42.50% | 42.29% | 5.65 pp | -104 | 48 | -2.17 |
| BTC Hourly | lstm | LSTM | 920 | 393 | 527 | 42.72% | 38.33% | 41.67% | 7.28 pp | -134 | 48 | -2.79 |
| BTC Hourly | xgb | XGBoost | 920 | 384 | 536 | 41.74% | 38.75% | 40.00% | 8.26 pp | -152 | 48 | -3.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 743 | 361 | 382 | 48.59% | 47.08% | 48.75% | 1.41 pp | -21 | 44 | -0.48 |
| BTC Daily | transformer | Transformer | 743 | 354 | 389 | 47.64% | 45.83% | 49.58% | 2.36 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 743 | 343 | 400 | 46.16% | 42.92% | 46.67% | 3.84 pp | -57 | 44 | -1.30 |
| BTC Daily | lstm | LSTM | 743 | 318 | 425 | 42.80% | 36.25% | 40.83% | 7.20 pp | -107 | 44 | -2.43 |
| BTC Daily | rf | RandomForest | 743 | 313 | 430 | 42.13% | 38.75% | 42.50% | 7.87 pp | -117 | 44 | -2.66 |
| BTC Daily | xgb | XGBoost | 753 | 297 | 456 | 39.44% | 36.25% | 37.71% | 10.56 pp | -159 | 44 | -3.61 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 515 | 249 | 266 | 48.35% | 45.42% | 48.33% | 1.65 pp | -17 | 49 | -0.35 |
| BTC Market Hours | transformer | Transformer | 515 | 247 | 268 | 47.96% | 47.08% | 48.54% | 2.04 pp | -21 | 49 | -0.43 |
| BTC Market Hours | nn | NN | 515 | 242 | 273 | 46.99% | 49.17% | 48.12% | 3.01 pp | -31 | 49 | -0.63 |
| BTC Market Hours | lstm | LSTM | 515 | 224 | 291 | 43.50% | 42.50% | 43.96% | 6.50 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 515 | 222 | 293 | 43.11% | 44.58% | 43.54% | 6.89 pp | -71 | 49 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 515 | 212 | 303 | 41.17% | 42.92% | 42.08% | 8.83 pp | -91 | 49 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 569 | 271 | 298 | 47.63% | 51.25% | 48.75% | 2.37 pp | -27 | 49 | -0.55 |
| BTC Market Hours Daily | nn | NN | 569 | 263 | 306 | 46.22% | 45.42% | 47.29% | 3.78 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 569 | 262 | 307 | 46.05% | 49.17% | 46.25% | 3.95 pp | -45 | 49 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 569 | 237 | 332 | 41.65% | 42.50% | 40.83% | 8.35 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 569 | 231 | 338 | 40.60% | 40.00% | 40.62% | 9.40 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | xgb | XGBoost | 569 | 227 | 342 | 39.89% | 40.83% | 38.96% | 10.11 pp | -115 | 49 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 161 | 74 | 87 | 45.96% | 45.96% | 45.96% | 4.04 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 74 | 87 | 45.96% | 45.96% | 45.96% | 4.04 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 71 | 90 | 44.10% | 44.10% | 44.10% | 5.90 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
