# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T20:16:35.794398+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 988 | 618 | 369 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 989 | 671 | 316 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 618 | 297 | 321 | 48.06% | 46.25% | 47.08% | 1.94 pp | -24 | 57 | -0.42 |
| BTC Market Hours | nn | NN | 618 | 296 | 322 | 47.90% | 50.42% | 49.79% | 2.10 pp | -26 | 57 | -0.46 |
| BTC Market Hours Daily | nn | NN | 671 | 317 | 354 | 47.24% | 50.42% | 48.75% | 2.76 pp | -37 | 57 | -0.65 |
| BTC Market Hours | transformer | Transformer | 618 | 288 | 330 | 46.60% | 46.25% | 45.42% | 3.40 pp | -42 | 57 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 671 | 314 | 357 | 46.80% | 49.17% | 47.29% | 3.20 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 671 | 311 | 360 | 46.35% | 47.92% | 47.71% | 3.65 pp | -49 | 57 | -0.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 846 | 401 | 445 | 47.40% | 43.33% | 45.83% | 2.60 pp | -44 | 48 | -0.92 |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1023 | 483 | 540 | 47.21% | 46.25% | 45.83% | 2.79 pp | -57 | 53 | -1.08 |
| BTC Daily | nn | NN | 846 | 395 | 451 | 46.69% | 45.83% | 45.21% | 3.31 pp | -56 | 48 | -1.17 |
| BTC Hourly | transformer | Transformer | 1023 | 480 | 543 | 46.92% | 46.67% | 45.00% | 3.08 pp | -63 | 53 | -1.19 |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| BTC Daily | transformer | Transformer | 846 | 390 | 456 | 46.10% | 38.33% | 44.38% | 3.90 pp | -66 | 48 | -1.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 618 | 265 | 353 | 42.88% | 42.92% | 43.33% | 7.12 pp | -88 | 57 | -1.54 |
| BTC Market Hours | rf | RandomForest | 618 | 265 | 353 | 42.88% | 43.75% | 42.08% | 7.12 pp | -88 | 57 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 618 | 263 | 355 | 42.56% | 46.67% | 42.71% | 7.44 pp | -92 | 57 | -1.61 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 671 | 279 | 392 | 41.58% | 43.75% | 41.67% | 8.42 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 671 | 275 | 396 | 40.98% | 43.75% | 41.04% | 9.02 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 671 | 275 | 396 | 40.98% | 43.75% | 41.25% | 9.02 pp | -121 | 57 | -2.12 |
| BTC Hourly | nn | NN | 1023 | 449 | 574 | 43.89% | 40.42% | 40.42% | 6.11 pp | -125 | 53 | -2.36 |
| BTC Hourly | rf | RandomForest | 1023 | 447 | 576 | 43.70% | 40.42% | 42.08% | 6.30 pp | -129 | 53 | -2.43 |
| BTC Daily | lstm | LSTM | 846 | 357 | 489 | 42.20% | 36.25% | 39.79% | 7.80 pp | -132 | 48 | -2.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| BTC Daily | rf | RandomForest | 846 | 352 | 494 | 41.61% | 37.50% | 40.83% | 8.39 pp | -142 | 48 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| BTC Hourly | lstm | LSTM | 1023 | 429 | 594 | 41.94% | 34.58% | 38.96% | 8.06 pp | -165 | 53 | -3.11 |
| BTC Hourly | xgb | XGBoost | 1023 | 420 | 603 | 41.06% | 35.00% | 37.71% | 8.94 pp | -183 | 53 | -3.45 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| BTC Daily | xgb | XGBoost | 856 | 338 | 518 | 39.49% | 37.92% | 36.25% | 10.51 pp | -180 | 48 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1023 | 483 | 540 | 47.21% | 46.25% | 45.83% | 2.79 pp | -57 | 53 | -1.08 |
| BTC Hourly | transformer | Transformer | 1023 | 480 | 543 | 46.92% | 46.67% | 45.00% | 3.08 pp | -63 | 53 | -1.19 |
| BTC Hourly | nn | NN | 1023 | 449 | 574 | 43.89% | 40.42% | 40.42% | 6.11 pp | -125 | 53 | -2.36 |
| BTC Hourly | rf | RandomForest | 1023 | 447 | 576 | 43.70% | 40.42% | 42.08% | 6.30 pp | -129 | 53 | -2.43 |
| BTC Hourly | lstm | LSTM | 1023 | 429 | 594 | 41.94% | 34.58% | 38.96% | 8.06 pp | -165 | 53 | -3.11 |
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
| BTC Market Hours Daily | nn | NN | 671 | 317 | 354 | 47.24% | 50.42% | 48.75% | 2.76 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 671 | 314 | 357 | 46.80% | 49.17% | 47.29% | 3.20 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 671 | 311 | 360 | 46.35% | 47.92% | 47.71% | 3.65 pp | -49 | 57 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 671 | 279 | 392 | 41.58% | 43.75% | 41.67% | 8.42 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 671 | 275 | 396 | 40.98% | 43.75% | 41.04% | 9.02 pp | -121 | 57 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 671 | 275 | 396 | 40.98% | 43.75% | 41.25% | 9.02 pp | -121 | 57 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
