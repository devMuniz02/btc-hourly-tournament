# BTC Model Metrics Report - All Rows

Generated at: 2026-09-08T11:36:54.578951+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1292 | 1004 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1167 | 802 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 887 | 564 | 322 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 889 | 618 | 269 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 564 | 274 | 290 | 48.58% | 47.08% | 47.71% | 1.42 pp | -16 | 53 | -0.30 |
| BTC Market Hours | nn | NN | 564 | 267 | 297 | 47.34% | 50.42% | 48.96% | 2.66 pp | -30 | 53 | -0.57 |
| BTC Market Hours | transformer | Transformer | 564 | 266 | 298 | 47.16% | 47.08% | 47.29% | 2.84 pp | -32 | 53 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 792 | 382 | 410 | 48.23% | 46.67% | 47.92% | 1.77 pp | -28 | 46 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 618 | 289 | 329 | 46.76% | 49.17% | 47.50% | 3.24 pp | -40 | 53 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 618 | 287 | 331 | 46.44% | 48.75% | 47.08% | 3.56 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | nn | NN | 618 | 287 | 331 | 46.44% | 46.67% | 47.71% | 3.56 pp | -44 | 53 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 970 | 460 | 510 | 47.42% | 48.75% | 46.04% | 2.58 pp | -50 | 50 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| BTC Daily | nn | NN | 792 | 368 | 424 | 46.46% | 45.42% | 45.00% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 792 | 368 | 424 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Hourly | transformer | Transformer | 970 | 452 | 518 | 46.60% | 44.17% | 43.75% | 3.40 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| BTC Market Hours | lstm | LSTM | 564 | 245 | 319 | 43.44% | 42.50% | 43.96% | 6.56 pp | -74 | 53 | -1.40 |
| BTC Market Hours | rf | RandomForest | 564 | 243 | 321 | 43.09% | 45.42% | 43.54% | 6.91 pp | -78 | 53 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 564 | 241 | 323 | 42.73% | 45.83% | 43.33% | 7.27 pp | -82 | 53 | -1.55 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | rf | RandomForest | 618 | 258 | 360 | 41.75% | 44.17% | 40.62% | 8.25 pp | -102 | 53 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 618 | 253 | 365 | 40.94% | 43.75% | 40.62% | 9.06 pp | -112 | 53 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 618 | 252 | 366 | 40.78% | 40.83% | 40.21% | 9.22 pp | -114 | 53 | -2.15 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| BTC Hourly | rf | RandomForest | 970 | 429 | 541 | 44.23% | 41.67% | 42.50% | 5.77 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 970 | 428 | 542 | 44.12% | 41.25% | 42.50% | 5.88 pp | -114 | 50 | -2.28 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 792 | 333 | 459 | 42.05% | 33.75% | 40.00% | 7.95 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 792 | 330 | 462 | 41.67% | 37.92% | 41.25% | 8.33 pp | -132 | 46 | -2.87 |
| BTC Hourly | lstm | LSTM | 970 | 412 | 558 | 42.47% | 37.50% | 41.04% | 7.53 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 970 | 400 | 570 | 41.24% | 36.25% | 38.54% | 8.76 pp | -170 | 50 | -3.40 |
| BTC Daily | xgb | XGBoost | 802 | 313 | 489 | 39.03% | 35.83% | 35.83% | 10.97 pp | -176 | 46 | -3.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 970 | 460 | 510 | 47.42% | 48.75% | 46.04% | 2.58 pp | -50 | 50 | -1.00 |
| BTC Hourly | transformer | Transformer | 970 | 452 | 518 | 46.60% | 44.17% | 43.75% | 3.40 pp | -66 | 50 | -1.32 |
| BTC Hourly | rf | RandomForest | 970 | 429 | 541 | 44.23% | 41.67% | 42.50% | 5.77 pp | -112 | 50 | -2.24 |
| BTC Hourly | nn | NN | 970 | 428 | 542 | 44.12% | 41.25% | 42.50% | 5.88 pp | -114 | 50 | -2.28 |
| BTC Hourly | lstm | LSTM | 970 | 412 | 558 | 42.47% | 37.50% | 41.04% | 7.53 pp | -146 | 50 | -2.92 |
| BTC Hourly | xgb | XGBoost | 970 | 400 | 570 | 41.24% | 36.25% | 38.54% | 8.76 pp | -170 | 50 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 792 | 382 | 410 | 48.23% | 46.67% | 47.92% | 1.77 pp | -28 | 46 | -0.61 |
| BTC Daily | nn | NN | 792 | 368 | 424 | 46.46% | 45.42% | 45.00% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 792 | 368 | 424 | 46.46% | 40.42% | 46.46% | 3.54 pp | -56 | 46 | -1.22 |
| BTC Daily | lstm | LSTM | 792 | 333 | 459 | 42.05% | 33.75% | 40.00% | 7.95 pp | -126 | 46 | -2.74 |
| BTC Daily | rf | RandomForest | 792 | 330 | 462 | 41.67% | 37.92% | 41.25% | 8.33 pp | -132 | 46 | -2.87 |
| BTC Daily | xgb | XGBoost | 802 | 313 | 489 | 39.03% | 35.83% | 35.83% | 10.97 pp | -176 | 46 | -3.83 |

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
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
