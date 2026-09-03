# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T16:23:39.473439+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1214 | 926 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1090 | 725 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 749 | 487 | 261 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 750 | 540 | 208 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 135 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 13:00:00+00:00 | 135 | 29 | 106 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 366 | 349 | 51.19% | 48.33% | 51.25% | 1.19 pp | 17 | 43 | 0.40 |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Daily | nn | NN | 715 | 340 | 375 | 47.55% | 46.67% | 48.54% | 2.45 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 540 | 250 | 290 | 46.30% | 49.17% | 47.29% | 3.70 pp | -40 | 47 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 892 | 425 | 467 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 892 | 423 | 469 | 47.42% | 48.75% | 47.29% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 540 | 247 | 293 | 45.74% | 47.50% | 46.88% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 540 | 247 | 293 | 45.74% | 44.17% | 46.67% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Daily | transformer | Transformer | 715 | 336 | 379 | 46.99% | 45.42% | 48.96% | 3.01 pp | -43 | 43 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| BTC Daily | lstm | LSTM | 715 | 322 | 393 | 45.03% | 38.33% | 44.17% | 4.97 pp | -71 | 43 | -1.65 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 715 | 319 | 396 | 44.62% | 41.67% | 44.38% | 5.38 pp | -77 | 43 | -1.79 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |
| BTC Hourly | nn | NN | 892 | 399 | 493 | 44.73% | 45.83% | 42.71% | 5.27 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 892 | 398 | 494 | 44.62% | 45.42% | 44.17% | 5.38 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 540 | 222 | 318 | 41.11% | 41.25% | 41.25% | 8.89 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 540 | 217 | 323 | 40.19% | 37.92% | 40.83% | 9.81 pp | -106 | 47 | -2.26 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 540 | 215 | 325 | 39.81% | 39.58% | 39.58% | 10.19 pp | -110 | 47 | -2.34 |
| BTC Hourly | lstm | LSTM | 892 | 382 | 510 | 42.83% | 39.17% | 42.08% | 7.17 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 892 | 377 | 515 | 42.26% | 42.92% | 42.08% | 7.74 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 725 | 291 | 434 | 40.14% | 36.67% | 39.17% | 9.86 pp | -143 | 43 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 892 | 425 | 467 | 47.65% | 50.00% | 48.12% | 2.35 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 892 | 423 | 469 | 47.42% | 48.75% | 47.29% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 892 | 399 | 493 | 44.73% | 45.83% | 42.71% | 5.27 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 892 | 398 | 494 | 44.62% | 45.42% | 44.17% | 5.38 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 892 | 382 | 510 | 42.83% | 39.17% | 42.08% | 7.17 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 892 | 377 | 515 | 42.26% | 42.92% | 42.08% | 7.74 pp | -138 | 47 | -2.94 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 715 | 366 | 349 | 51.19% | 48.33% | 51.25% | 1.19 pp | 17 | 43 | 0.40 |
| BTC Daily | nn | NN | 715 | 340 | 375 | 47.55% | 46.67% | 48.54% | 2.45 pp | -35 | 43 | -0.81 |
| BTC Daily | transformer | Transformer | 715 | 336 | 379 | 46.99% | 45.42% | 48.96% | 3.01 pp | -43 | 43 | -1.00 |
| BTC Daily | lstm | LSTM | 715 | 322 | 393 | 45.03% | 38.33% | 44.17% | 4.97 pp | -71 | 43 | -1.65 |
| BTC Daily | rf | RandomForest | 715 | 319 | 396 | 44.62% | 41.67% | 44.38% | 5.38 pp | -77 | 43 | -1.79 |
| BTC Daily | xgb | XGBoost | 725 | 291 | 434 | 40.14% | 36.67% | 39.17% | 9.86 pp | -143 | 43 | -3.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 487 | 234 | 253 | 48.05% | 43.75% | 48.12% | 1.95 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 487 | 229 | 258 | 47.02% | 48.75% | 47.29% | 2.98 pp | -29 | 47 | -0.62 |
| BTC Market Hours | transformer | Transformer | 487 | 227 | 260 | 46.61% | 42.50% | 46.88% | 3.39 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 487 | 210 | 277 | 43.12% | 41.25% | 43.12% | 6.88 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 487 | 208 | 279 | 42.71% | 41.67% | 42.92% | 7.29 pp | -71 | 47 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 487 | 197 | 290 | 40.45% | 39.17% | 40.42% | 9.55 pp | -93 | 47 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 540 | 250 | 290 | 46.30% | 49.17% | 47.29% | 3.70 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 540 | 247 | 293 | 45.74% | 47.50% | 46.88% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 540 | 247 | 293 | 45.74% | 44.17% | 46.67% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 540 | 222 | 318 | 41.11% | 41.25% | 41.25% | 8.89 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 540 | 217 | 323 | 40.19% | 37.92% | 40.83% | 9.81 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 540 | 215 | 325 | 39.81% | 39.58% | 39.58% | 10.19 pp | -110 | 47 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 70 | 65 | 51.85% | 51.85% | 51.85% | 1.85 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 64 | 71 | 47.41% | 47.41% | 47.41% | 2.59 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
