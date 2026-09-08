# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T10:25:29.350719+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1291 | 1003 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1167 | 802 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 887 | 564 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 889 | 618 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 564 | 274 | 290 | 48.58% | 47.08% | 47.71% | 1.42 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 564 | 267 | 297 | 47.34% | 50.42% | 48.96% | 2.66 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 564 | 266 | 298 | 47.16% | 47.08% | 47.29% | 2.84 pp | -32 | 53 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 792 | 382 | 410 | 48.23% | 46.67% | 47.92% | 1.77 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 618 | 289 | 329 | 46.76% | 49.17% | 47.50% | 3.24 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 618 | 287 | 331 | 46.44% | 48.75% | 47.08% | 3.56 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | nn | NN | 618 | 287 | 331 | 46.44% | 46.67% | 47.71% | 3.56 pp | -44 | 53 | -0.83 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 969 | 460 | 509 | 47.47% | 48.75% | 46.25% | 2.53 pp | -49 | 50 | -0.98 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 792 | 369 | 423 | 46.59% | 45.83% | 45.21% | 3.41 pp | -54 | 46 | -1.17 |
| BTC Daily | transformer | Transformer | 792 | 369 | 423 | 46.59% | 40.83% | 46.67% | 3.41 pp | -54 | 46 | -1.17 |
| BTC Hourly | transformer | Transformer | 969 | 451 | 518 | 46.54% | 43.75% | 43.54% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Market Hours | lstm | LSTM | 564 | 245 | 319 | 43.44% | 42.50% | 43.96% | 6.56 pp | -74 | 53 | -1.40 |
| BTC Market Hours | rf | RandomForest | 564 | 243 | 321 | 43.09% | 45.42% | 43.54% | 6.91 pp | -78 | 53 | -1.47 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 564 | 241 | 323 | 42.73% | 45.83% | 43.33% | 7.27 pp | -82 | 53 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 618 | 258 | 360 | 41.75% | 44.17% | 40.62% | 8.25 pp | -102 | 53 | -1.92 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 618 | 253 | 365 | 40.94% | 43.75% | 40.62% | 9.06 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 618 | 252 | 366 | 40.78% | 40.83% | 40.21% | 9.22 pp | -114 | 53 | -2.15 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Hourly | nn | NN | 969 | 428 | 541 | 44.17% | 41.25% | 42.71% | 5.83 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 969 | 428 | 541 | 44.17% | 41.25% | 42.50% | 5.83 pp | -113 | 50 | -2.26 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Daily | lstm | LSTM | 792 | 333 | 459 | 42.05% | 33.75% | 40.00% | 7.95 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 792 | 331 | 461 | 41.79% | 38.33% | 41.46% | 8.21 pp | -130 | 46 | -2.83 |
| BTC Hourly | lstm | LSTM | 969 | 412 | 557 | 42.52% | 37.50% | 41.25% | 7.48 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 969 | 399 | 570 | 41.18% | 35.83% | 38.54% | 8.82 pp | -171 | 50 | -3.42 |
| BTC Daily | xgb | XGBoost | 802 | 314 | 488 | 39.15% | 36.25% | 36.04% | 10.85 pp | -174 | 46 | -3.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 969 | 460 | 509 | 47.47% | 48.75% | 46.25% | 2.53 pp | -49 | 50 | -0.98 |
| BTC Hourly | transformer | Transformer | 969 | 451 | 518 | 46.54% | 43.75% | 43.54% | 3.46 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 969 | 428 | 541 | 44.17% | 41.25% | 42.71% | 5.83 pp | -113 | 50 | -2.26 |
| BTC Hourly | rf | RandomForest | 969 | 428 | 541 | 44.17% | 41.25% | 42.50% | 5.83 pp | -113 | 50 | -2.26 |
| BTC Hourly | lstm | LSTM | 969 | 412 | 557 | 42.52% | 37.50% | 41.25% | 7.48 pp | -145 | 50 | -2.90 |
| BTC Hourly | xgb | XGBoost | 969 | 399 | 570 | 41.18% | 35.83% | 38.54% | 8.82 pp | -171 | 50 | -3.42 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 792 | 382 | 410 | 48.23% | 46.67% | 47.92% | 1.77 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 792 | 369 | 423 | 46.59% | 45.83% | 45.21% | 3.41 pp | -54 | 46 | -1.17 |
| BTC Daily | transformer | Transformer | 792 | 369 | 423 | 46.59% | 40.83% | 46.67% | 3.41 pp | -54 | 46 | -1.17 |
| BTC Daily | lstm | LSTM | 792 | 333 | 459 | 42.05% | 33.75% | 40.00% | 7.95 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 792 | 331 | 461 | 41.79% | 38.33% | 41.46% | 8.21 pp | -130 | 46 | -2.83 |
| BTC Daily | xgb | XGBoost | 802 | 314 | 488 | 39.15% | 36.25% | 36.04% | 10.85 pp | -174 | 46 | -3.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 564 | 274 | 290 | 48.58% | 47.08% | 47.71% | 1.42 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 564 | 267 | 297 | 47.34% | 50.42% | 48.96% | 2.66 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 564 | 266 | 298 | 47.16% | 47.08% | 47.29% | 2.84 pp | -32 | 53 | -0.60 |
| BTC Market Hours | lstm | LSTM | 564 | 245 | 319 | 43.44% | 42.50% | 43.96% | 6.56 pp | -74 | 53 | -1.40 |
| BTC Market Hours | rf | RandomForest | 564 | 243 | 321 | 43.09% | 45.42% | 43.54% | 6.91 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 564 | 241 | 323 | 42.73% | 45.83% | 43.33% | 7.27 pp | -82 | 53 | -1.55 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 618 | 289 | 329 | 46.76% | 49.17% | 47.50% | 3.24 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 618 | 287 | 331 | 46.44% | 48.75% | 47.08% | 3.56 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | nn | NN | 618 | 287 | 331 | 46.44% | 46.67% | 47.71% | 3.56 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 618 | 258 | 360 | 41.75% | 44.17% | 40.62% | 8.25 pp | -102 | 53 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 618 | 253 | 365 | 40.94% | 43.75% | 40.62% | 9.06 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 618 | 252 | 366 | 40.78% | 40.83% | 40.21% | 9.22 pp | -114 | 53 | -2.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
