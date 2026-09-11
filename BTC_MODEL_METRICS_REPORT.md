# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T21:32:06.978089+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1345 | 1057 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1221 | 856 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 20:00:00+00:00 | 989 | 618 | 370 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 20:00:00+00:00 | 991 | 672 | 317 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 618 | 297 | 321 | 48.06% | 46.25% | 47.08% | 1.94 pp | -24 | 57 | -0.42 |
| BTC Market Hours | nn | NN | 618 | 296 | 322 | 47.90% | 50.42% | 49.79% | 2.10 pp | -26 | 57 | -0.46 |
| BTC Market Hours Daily | nn | NN | 672 | 318 | 354 | 47.32% | 50.83% | 48.75% | 2.68 pp | -36 | 57 | -0.63 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 672 | 315 | 357 | 46.88% | 49.17% | 47.29% | 3.12 pp | -42 | 57 | -0.74 |
| BTC Market Hours | transformer | Transformer | 618 | 288 | 330 | 46.60% | 46.25% | 45.42% | 3.40 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 672 | 312 | 360 | 46.43% | 47.92% | 47.71% | 3.57 pp | -48 | 57 | -0.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 846 | 401 | 445 | 47.40% | 43.33% | 45.83% | 2.60 pp | -44 | 48 | -0.92 |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1023 | 483 | 540 | 47.21% | 46.25% | 45.83% | 2.79 pp | -57 | 53 | -1.08 |
| BTC Hourly | transformer | Transformer | 1023 | 481 | 542 | 47.02% | 47.08% | 45.21% | 2.98 pp | -61 | 53 | -1.15 |
| BTC Daily | nn | NN | 846 | 395 | 451 | 46.69% | 45.83% | 45.21% | 3.31 pp | -56 | 48 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 846 | 390 | 456 | 46.10% | 38.33% | 44.38% | 3.90 pp | -66 | 48 | -1.38 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | lstm | LSTM | 618 | 265 | 353 | 42.88% | 42.92% | 43.33% | 7.12 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 618 | 265 | 353 | 42.88% | 43.75% | 42.08% | 7.12 pp | -88 | 57 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 618 | 263 | 355 | 42.56% | 46.67% | 42.71% | 7.44 pp | -92 | 57 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 672 | 279 | 393 | 41.52% | 43.33% | 41.46% | 8.48 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 672 | 276 | 396 | 41.07% | 44.17% | 41.04% | 8.93 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 672 | 275 | 397 | 40.92% | 43.75% | 41.25% | 9.08 pp | -122 | 57 | -2.14 |
| BTC Hourly | nn | NN | 1023 | 450 | 573 | 43.99% | 40.42% | 40.62% | 6.01 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1023 | 447 | 576 | 43.70% | 40.42% | 42.08% | 6.30 pp | -129 | 53 | -2.43 |
| BTC Daily | lstm | LSTM | 846 | 357 | 489 | 42.20% | 36.25% | 39.79% | 7.80 pp | -132 | 48 | -2.75 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 846 | 352 | 494 | 41.61% | 37.50% | 40.83% | 8.39 pp | -142 | 48 | -2.96 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| BTC Hourly | lstm | LSTM | 1023 | 429 | 594 | 41.94% | 34.17% | 38.96% | 8.06 pp | -165 | 53 | -3.11 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1023 | 420 | 603 | 41.06% | 35.00% | 37.71% | 8.94 pp | -183 | 53 | -3.45 |
| BTC Daily | xgb | XGBoost | 856 | 338 | 518 | 39.49% | 37.92% | 36.25% | 10.51 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1023 | 483 | 540 | 47.21% | 46.25% | 45.83% | 2.79 pp | -57 | 53 | -1.08 |
| BTC Hourly | transformer | Transformer | 1023 | 481 | 542 | 47.02% | 47.08% | 45.21% | 2.98 pp | -61 | 53 | -1.15 |
| BTC Hourly | nn | NN | 1023 | 450 | 573 | 43.99% | 40.42% | 40.62% | 6.01 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1023 | 447 | 576 | 43.70% | 40.42% | 42.08% | 6.30 pp | -129 | 53 | -2.43 |
| BTC Hourly | lstm | LSTM | 1023 | 429 | 594 | 41.94% | 34.17% | 38.96% | 8.06 pp | -165 | 53 | -3.11 |
| BTC Hourly | xgb | XGBoost | 1023 | 420 | 603 | 41.06% | 35.00% | 37.71% | 8.94 pp | -183 | 53 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 846 | 401 | 445 | 47.40% | 43.33% | 45.83% | 2.60 pp | -44 | 48 | -0.92 |
| BTC Daily | nn | NN | 846 | 395 | 451 | 46.69% | 45.83% | 45.21% | 3.31 pp | -56 | 48 | -1.17 |
| BTC Daily | transformer | Transformer | 846 | 390 | 456 | 46.10% | 38.33% | 44.38% | 3.90 pp | -66 | 48 | -1.38 |
| BTC Daily | lstm | LSTM | 846 | 357 | 489 | 42.20% | 36.25% | 39.79% | 7.80 pp | -132 | 48 | -2.75 |
| BTC Daily | rf | RandomForest | 846 | 352 | 494 | 41.61% | 37.50% | 40.83% | 8.39 pp | -142 | 48 | -2.96 |
| BTC Daily | xgb | XGBoost | 856 | 338 | 518 | 39.49% | 37.92% | 36.25% | 10.51 pp | -180 | 48 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 618 | 297 | 321 | 48.06% | 46.25% | 47.08% | 1.94 pp | -24 | 57 | -0.42 |
| BTC Market Hours | nn | NN | 618 | 296 | 322 | 47.90% | 50.42% | 49.79% | 2.10 pp | -26 | 57 | -0.46 |
| BTC Market Hours | transformer | Transformer | 618 | 288 | 330 | 46.60% | 46.25% | 45.42% | 3.40 pp | -42 | 57 | -0.74 |
| BTC Market Hours | lstm | LSTM | 618 | 265 | 353 | 42.88% | 42.92% | 43.33% | 7.12 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 618 | 265 | 353 | 42.88% | 43.75% | 42.08% | 7.12 pp | -88 | 57 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 618 | 263 | 355 | 42.56% | 46.67% | 42.71% | 7.44 pp | -92 | 57 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 672 | 318 | 354 | 47.32% | 50.83% | 48.75% | 2.68 pp | -36 | 57 | -0.63 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 672 | 315 | 357 | 46.88% | 49.17% | 47.29% | 3.12 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 672 | 312 | 360 | 46.43% | 47.92% | 47.71% | 3.57 pp | -48 | 57 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 672 | 279 | 393 | 41.52% | 43.33% | 41.46% | 8.48 pp | -114 | 57 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 672 | 276 | 396 | 41.07% | 44.17% | 41.04% | 8.93 pp | -120 | 57 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 672 | 275 | 397 | 40.92% | 43.75% | 41.25% | 9.08 pp | -122 | 57 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
